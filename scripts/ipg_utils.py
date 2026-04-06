from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from opt_einsum import contract
from scipy.linalg import expm, logm

from pyquda import init
from pyquda.field import LatticeGauge
from pyquda_utils import core, gamma, io, source

_PYQUDA_INITIALIZED = False
_SU2_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
_Z3_PHASES: Tuple[complex, ...] = (
    np.exp(0j),
    np.exp(2j * np.pi / 3),
    np.exp(-2j * np.pi / 3),
)


@dataclass(frozen=True)
class PropagatorParams:
    latt_size: Tuple[int, int, int, int] = (16, 16, 16, 16)
    t_boundary: int = -1
    xi_0: float = 1.0
    nu: float = 1.0
    mass: float = -0.038888
    tol: float = 1e-8
    maxiter: int = 10000
    csw_r: float = 1.02868
    csw_t: float = 1.02868
    multigrid: Optional[Sequence[Sequence[int]]] = None

    @property
    def anisotropy(self) -> float:
        return self.xi_0 / self.nu


@dataclass
class IPGResult:
    gauge: LatticeGauge
    transform: np.ndarray
    averaged_temporal_links: np.ndarray
    projected_temporal_links: np.ndarray
    polyakov_matrix: np.ndarray
    target_matrix: np.ndarray
    logm_error: float
    boundary_error: float
    transformed_spread: float


def determinant_error(matrix: np.ndarray) -> float:
    return float(abs(np.linalg.det(matrix) - 1.0))


def ensure_pyquda_initialized(resource_path: Optional[Path] = None) -> Path:
    global _PYQUDA_INITIALIZED
    if resource_path is None:
        resource_path = Path(__file__).resolve().parent / ".cache"
    resource_path = resource_path.resolve()
    resource_path.mkdir(parents=True, exist_ok=True)
    if not _PYQUDA_INITIALIZED:
        init([1, 1, 1, 1], resource_path=str(resource_path))
        _PYQUDA_INITIALIZED = True
    return resource_path


def extract_cfg_index(path: Path) -> int:
    dot_tokens = path.name.split(".")
    for token in reversed(dot_tokens):
        if token.isdigit():
            return int(token)
    match = re.search(r"\.(\d+)$", path.name)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"Could not extract configuration index from {path}")


def list_gauge_files(
    input_dir: Path,
    pattern: str,
    cfg_start: Optional[int] = None,
    cfg_stop: Optional[int] = None,
) -> List[Path]:
    files = [path for path in input_dir.glob(pattern) if path.is_file()]
    selected: List[Path] = []
    for path in files:
        cfg = extract_cfg_index(path)
        if cfg_start is not None and cfg < cfg_start:
            continue
        if cfg_stop is not None and cfg > cfg_stop:
            continue
        selected.append(path)
    return sorted(selected, key=lambda path: (extract_cfg_index(path), path.name))


def default_ipg_name(input_name: str) -> str:
    if ".cg." in input_name:
        return input_name.replace(".cg.", ".cg.ipg.", 1)
    tolerance_suffix = re.search(r"(\.\d+e[+-]?\d+)$", input_name)
    if tolerance_suffix is not None:
        return f"{input_name[:tolerance_suffix.start()]}{'.ipg'}{tolerance_suffix.group(1)}"
    return f"{input_name}.ipg"


def parse_cfg_list(cfg_list: Optional[str]) -> Optional[List[int]]:
    if not cfg_list:
        return None
    items = [item.strip() for item in cfg_list.split(",") if item.strip()]
    return sorted(int(item) for item in items)


def normalize_to_su3(matrix: np.ndarray) -> np.ndarray:
    det = np.linalg.det(matrix)
    if abs(det) == 0:
        raise ValueError("Cannot normalize a singular matrix to SU(3)")
    return matrix / np.power(det, 1.0 / 3.0)


def unitary_error(matrix: np.ndarray) -> float:
    ident = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - ident))


def polar_project_su3(matrix: np.ndarray) -> np.ndarray:
    left, _, right_h = np.linalg.svd(np.asarray(matrix, dtype=np.complex128))
    return normalize_to_su3(left @ right_h)


def _optimal_su2_update(current: np.ndarray, target: np.ndarray, pair: Tuple[int, int]) -> np.ndarray:
    objective = current @ target.conj().T
    sub = objective[np.ix_(pair, pair)]
    left, _, right_h = np.linalg.svd(sub)
    update = right_h.conj().T @ left.conj().T
    det = np.linalg.det(update)
    if abs(det) == 0:
        raise ValueError("Encountered singular SU(2) update during Cabibbo-Marinari projection")
    update = update / np.sqrt(det)
    embedded = np.eye(3, dtype=np.complex128)
    embedded[np.ix_(pair, pair)] = update
    return embedded


def cabibbo_marinari_project_su3(
    matrix: np.ndarray,
    max_sweeps: int = 64,
    tol: float = 1e-14,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    current = np.eye(3, dtype=np.complex128)
    for _ in range(max_sweeps):
        previous = current.copy()
        for pair in _SU2_PAIRS:
            current = _optimal_su2_update(current, matrix, pair) @ current
        if np.linalg.norm(current - previous) < tol:
            break
    current = normalize_to_su3(current)
    if unitary_error(current) > 1e-10:
        raise RuntimeError("Cabibbo-Marinari projection failed to produce a unitary SU(3) matrix")
    return current


def project_to_su3(matrix: np.ndarray, method: str = "cabibbo-marinari") -> np.ndarray:
    method = method.lower()
    if method in {"cabibbo-marinari", "cm", "cabibbo_marinari"}:
        return cabibbo_marinari_project_su3(matrix)
    if method == "polar":
        return polar_project_su3(matrix)
    raise ValueError(f"Unsupported SU(3) projection method: {method}")


def z3_align(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return ``phase * matrix`` for the Z_3 center phase minimising distance to *reference*."""
    return min(
        (phase * matrix for phase in _Z3_PHASES),
        key=lambda candidate: float(np.linalg.norm(candidate - reference)),
    )


def average_temporal_links(gauge_lexico: np.ndarray) -> np.ndarray:
    return gauge_lexico[3].mean(axis=(1, 2, 3)) # shape of gauge_lexico is (Nd, T, Lz, Ly, Lx, 3, 3), we take the T direction gauge links and average over spatial dimensions to get (T, 3, 3)


def projected_temporal_links(
    gauge_lexico: np.ndarray,
    projection_method: str = "cabibbo-marinari",
    z3_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    averaged = average_temporal_links(gauge_lexico)
    projected = np.stack([project_to_su3(link, method=projection_method) for link in averaged], axis=0)
    if z3_reference is not None:
        projected = np.stack([z3_align(link, z3_reference) for link in projected], axis=0)
    return projected


def integrated_polyakov_matrix(projected_temporal_links: np.ndarray) -> np.ndarray:
    polyakov = np.eye(3, dtype=np.complex128)
    for link in projected_temporal_links:
        polyakov = polyakov @ link
    return normalize_to_su3(polyakov)


def _build_transform_from_target(projected_temporal_links: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    extent_t = projected_temporal_links.shape[0]
    transform = np.empty((extent_t, 3, 3), dtype=np.complex128)
    transform[0] = np.eye(3, dtype=np.complex128)
    for t in range(extent_t - 1):
        transform[t + 1] = target.conj().T @ transform[t] @ projected_temporal_links[t]
    boundary = target.conj().T @ transform[-1] @ projected_temporal_links[-1]
    boundary_error = float(np.linalg.norm(boundary - np.eye(3, dtype=np.complex128)))
    return transform, boundary_error


def matrix_t_root_su3(
    polyakov: np.ndarray,
    extent_t: int,
    projection_method: str,
    projected_temporal_links: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    log_polyakov, logm_error = logm(polyakov, disp=False)
    root = expm(log_polyakov / extent_t)
    root = project_to_su3(root, method=projection_method)
    if projected_temporal_links is not None:
        center_phases = (
            np.exp(0j),
            np.exp(2j * np.pi / 3),
            np.exp(-2j * np.pi / 3),
        )
        candidates = []
        for phase in center_phases:
            candidate = phase * root
            transform, boundary_error = _build_transform_from_target(projected_temporal_links, candidate)
            candidates.append((boundary_error, candidate))
        _, root = min(candidates, key=lambda item: item[0])
    return root, float(logm_error)


def build_ipg_transform(
    gauge_lexico: np.ndarray,
    projection_method: str = "cabibbo-marinari",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    averaged = average_temporal_links(gauge_lexico)
    projected = projected_temporal_links(gauge_lexico, projection_method=projection_method)
    polyakov = integrated_polyakov_matrix(projected)
    extent_t = gauge_lexico.shape[1]
    target, logm_error = matrix_t_root_su3(polyakov, extent_t, projection_method, projected_temporal_links=projected)
    transform, boundary_error = _build_transform_from_target(projected, target)
    return transform, averaged, projected, target, logm_error, boundary_error


def apply_residual_transform_lexico(gauge_lexico: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transformed = gauge_lexico.copy()
    extent_t = gauge_lexico.shape[1]
    for t in range(extent_t):
        left = transform[t]
        right_same = transform[t].conj().T
        right_next = transform[(t + 1) % extent_t].conj().T
        for mu in range(3):
            transformed[mu, t] = left @ gauge_lexico[mu, t] @ right_same
        transformed[3, t] = left @ gauge_lexico[3, t] @ right_next
    return transformed


def projected_temporal_spread(
    gauge_lexico: np.ndarray,
    projection_method: str = "cabibbo-marinari",
    z3_reference: Optional[np.ndarray] = None,
) -> float:
    projected = projected_temporal_links(gauge_lexico, projection_method=projection_method, z3_reference=z3_reference)
    return float(max(np.linalg.norm(link - projected[0]) for link in projected))


def target_residual_error(projected_temporal_links: np.ndarray, transform: np.ndarray, target: np.ndarray) -> float:
    extent_t = projected_temporal_links.shape[0]
    return float(
        max(
            np.linalg.norm(transform[t] @ projected_temporal_links[t] @ transform[(t + 1) % extent_t].conj().T - target)
            for t in range(extent_t)
        )
    )


def target_deviation(projected_temporal_links: np.ndarray, target: np.ndarray) -> float:
    return float(max(np.linalg.norm(z3_align(link, target) - target) for link in projected_temporal_links))


def transform_unitarity_error(transform: np.ndarray) -> float:
    return float(max(unitary_error(matrix) for matrix in transform))


def transform_determinant_error(transform: np.ndarray) -> float:
    return float(max(determinant_error(matrix) for matrix in transform))


def gauge_max_abs_diff(lhs_lexico: np.ndarray, rhs_lexico: np.ndarray) -> float:
    return float(np.max(np.abs(lhs_lexico - rhs_lexico)))


def read_nersc_gauge(path: Path) -> LatticeGauge:
    return io.readNERSCGauge(str(path), checksum=False, plaquette=False, link_trace=False)


def write_nersc_gauge(path: Path, gauge: LatticeGauge) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.writeNERSCGauge(str(path), gauge)


def make_ipg_gauge(
    gauge: LatticeGauge,
    projection_method: str = "cabibbo-marinari",
) -> IPGResult:
    lexico = gauge.lexico().copy()
    transform, averaged, projected, target, logm_error, boundary_error = build_ipg_transform(
        lexico, projection_method=projection_method
    )
    transformed_lexico = apply_residual_transform_lexico(lexico, transform)
    transformed_gauge = LatticeGauge(gauge.latt_info, gauge.latt_info.evenodd(transformed_lexico, True))
    spread = projected_temporal_spread(transformed_gauge.lexico(), projection_method="polar", z3_reference=target)
    return IPGResult(
        gauge=transformed_gauge,
        transform=transform,
        averaged_temporal_links=averaged,
        projected_temporal_links=projected,
        polyakov_matrix=integrated_polyakov_matrix(projected),
        target_matrix=target,
        logm_error=logm_error,
        boundary_error=boundary_error,
        transformed_spread=spread,
    )


def build_dirac(params: PropagatorParams):
    latt_info = core.LatticeInfo(list(params.latt_size), params.t_boundary, params.anisotropy)
    dirac = core.getClover(
        latt_info,
        params.mass,
        params.tol,
        params.maxiter,
        params.xi_0,
        params.csw_r,
        params.csw_t,
        params.multigrid,
    )
    return latt_info, dirac


def current_spatial_line_observable_with_dirac(gauge: LatticeGauge, latt_info, dirac) -> np.ndarray:
    gamma0 = gamma.gamma(0)
    with dirac.useGauge(gauge):
        point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
        point_propag = core.invertPropagator(dirac, point_source)
        contracted = contract("wtzyxijaa,ji->wtzyx", point_propag.data, gamma0).real.get()
        gathered = core.gatherLattice(core.lexico(contracted, [0, 1, 2, 3, 4]), [0, 1, 2, 3])
    return np.asarray(gathered[0, 0, :, 0], dtype=np.float64)


def current_spatial_line_observable(gauge: LatticeGauge, params: PropagatorParams) -> np.ndarray:
    latt_info, dirac = build_dirac(params)
    return current_spatial_line_observable_with_dirac(gauge, latt_info, dirac)


def compare_observables(lhs: np.ndarray, rhs: np.ndarray) -> Tuple[float, float]:
    diff = lhs - rhs
    max_abs = float(np.max(np.abs(diff)))
    rel = float(np.linalg.norm(diff) / np.linalg.norm(lhs))
    return max_abs, rel


def cfg_subset_from_args(files: Sequence[Path], cfg_list: Optional[Iterable[int]], n_conf: Optional[int]) -> List[Path]:
    selected = list(files)
    if cfg_list is not None:
        wanted = set(cfg_list)
        selected = [path for path in selected if extract_cfg_index(path) in wanted]
    if n_conf is not None:
        selected = selected[:n_conf]
    return selected
