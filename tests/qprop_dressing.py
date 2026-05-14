# %%
import os
import gvar as gv
from pyquda import init
from pyquda_utils import core, io, source, gamma
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *


if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")


ensemble = "S16T16_cg_ipg"  # "S16T16", "S16T16_cg", "S16T16_cg_ipg", "S32T32_cg_ipg"


init([1, 1, 1, 1], resource_path=".cache")
N_conf = 50  # Number of configurations to process

# Lattice parameters
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None # [[4, 4, 4, 4], [2, 2, 2, 8]]

latt_size = [16, 16, 16, 16]
latt_info = core.LatticeInfo(latt_size, -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)
is_root = latt_info.mpi_rank == 0

I = gamma.gamma(0)
gX = gamma.gamma(1)
gY = gamma.gamma(2)
gZ = gamma.gamma(4)
gT = gamma.gamma(8)

gamma_ops = [("I", I), ("gX", gX), ("gY", gY), ("gZ", gZ), ("gT", gT)]

# 4-momenta: spatial × temporal (temporal up to T/2 for better pole-mass fit)
spatial_momenta = [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]]
temporal_momenta = [0, 2, 4, 6, 8]
momentum_list = [[px, py, pz, pt] for px, py, pz in spatial_momenta for pt in temporal_momenta]
momentum_label = [f"({px},{py},{pz},{pt})" for px, py, pz, pt in momentum_list]
momentum_array = np.asarray(momentum_list, dtype=np.float64)
mom_phase = MomentumPhase(latt_info)

lattice_spacing_fm = 0.11
hbarc_GeV_fm = 0.1973269804
latt_extent = np.asarray(latt_size, dtype=np.float64)
momentum_angles = 2 * np.pi * momentum_array / latt_extent
p_phys_mu = momentum_angles / lattice_spacing_fm * hbarc_GeV_fm
k_mu = np.sin(momentum_angles)
a_GeV_inv = lattice_spacing_fm / hbarc_GeV_fm
Nc = 3
Ns = 4


def dressing_from_greens(greens_by_gamma):
    coeff_norm = Nc * Ns
    c_m = greens_by_gamma["I"] / coeff_norm
    c_x = greens_by_gamma["gX"] / coeff_norm
    c_y = greens_by_gamma["gY"] / coeff_norm
    c_z = greens_by_gamma["gZ"] / coeff_norm
    c_t = greens_by_gamma["gT"] / coeff_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = c_m**2 - c_x**2 - c_y**2 - c_z**2 - c_t**2
        coeff_inv_x = -c_x / denom
        coeff_inv_y = -c_y / denom
        coeff_inv_z = -c_z / denom
        coeff_inv_t = -c_t / denom

        As_components = np.stack(
            [
                np.where(k_mu[:, 0] != 0, 1j * coeff_inv_x / (a_GeV_inv * k_mu[:, 0]), np.nan),
                np.where(k_mu[:, 1] != 0, 1j * coeff_inv_y / (a_GeV_inv * k_mu[:, 1]), np.nan),
                np.where(k_mu[:, 2] != 0, 1j * coeff_inv_z / (a_GeV_inv * k_mu[:, 2]), np.nan),
            ]
        )
        As_valid = ~np.isnan(As_components)
        As_count = np.sum(As_valid, axis=0)
        As = np.full(c_m.shape, np.nan, dtype=np.complex128)
        As_sum = np.nansum(As_components, axis=0)
        As[As_count > 0] = As_sum[As_count > 0] / As_count[As_count > 0]
        At = np.where(k_mu[:, 3] != 0, 1j * coeff_inv_t / (a_GeV_inv * k_mu[:, 3]), np.nan)
        Bm = c_m / denom
        M = Bm / As

    return {"As": As, "At": At, "Bm": Bm, "M": M}


N_src = 4
rng = np.random.RandomState(42)
if is_root:
    print(f"Using {N_src} random source positions per config, temporal momenta n4 = {temporal_momenta}")

# Accumulate Eq. 20 inverse-propagator dressing functions across configurations.
point_quark_dressing = {"As": [], "At": [], "Bm": [], "M": []}

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):

    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")
    elif ensemble == "S32T32_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S32T32_cg_ipg/gauge/wilson_b5_95_fixed.{cfg}.ipg")

    # gauge.stoutSmear(1, 0.125, 4)

    # source_positions = rng.randint(0, latt_size, size=(N_src, 4)).tolist()
    source_positions = [[0, 0, 0, 0]] # todo

    with dirac.useGauge(gauge):
        cfg_greens_by_gamma = {name: [] for name, _ in gamma_ops}

        for src_position in source_positions:
            # Phase kernel e^{ip·(x - y)} with automatic source-position correction
            momentum_phases = mom_phase.getPhases(momentum_list, x0=src_position)

            point_source = source.propagator(latt_info, "point", src_position)
            point_propag = core.invertPropagator(dirac, point_source)

            for gamma_name, gamma_matrix in gamma_ops:
                point_quark_green = contract(
                    "pwtzyx,wtzyxijaa,ji->p",
                    momentum_phases,
                    point_propag.data,
                    gamma_matrix,
                ).get()

                if is_root:
                    cfg_greens_by_gamma[gamma_name].append(point_quark_green)

        if is_root:
            cfg_mean_by_gamma = {
                gamma_name: np.mean(cfg_greens_by_gamma[gamma_name], axis=0)
                for gamma_name in cfg_greens_by_gamma
            }
            cfg_dressing = dressing_from_greens(cfg_mean_by_gamma)
            for dressing_name in point_quark_dressing:
                point_quark_dressing[dressing_name].append(cfg_dressing[dressing_name])


# %%
if is_root:
    # Cache complex Eq. 20 dressing functions.
    cache_dict = dict(
        momentum_list=np.asarray(momentum_list),
        momentum_label=np.asarray(momentum_label),
        latt_size=np.asarray(latt_size),
        lattice_spacing_fm=np.asarray(lattice_spacing_fm),
        p_phys_mu=p_phys_mu,
        k_mu=k_mu,
    )
    for dressing_name in point_quark_dressing:
        cache_dict[dressing_name] = np.asarray(point_quark_dressing[dressing_name])

    os.makedirs("artifacts/data", exist_ok=True)
    np.savez(f"artifacts/data/qprop_dressing_{ensemble}.npz", **cache_dict)
    print(f"Cached to artifacts/data/qprop_dressing_{ensemble}.npz")

    os.makedirs("artifacts/plots", exist_ok=True)
    for dressing_name in point_quark_dressing:
        dressing = np.asarray(point_quark_dressing[dressing_name])
        dressing_re = np.real(dressing)
        dressing_im = np.imag(dressing)
        dressing_norm = np.abs(dressing)

        dressing_re_jk_avg = jk_ls_avg(jackknife(dressing_re))
        dressing_im_jk_avg = jk_ls_avg(jackknife(dressing_im))
        dressing_norm_jk_avg = jk_ls_avg(jackknife(dressing_norm))

        x_values = np.arange(len(momentum_label))

        fig, ax = default_plot()
        ax.errorbar(
            x_values,
            gv.mean(dressing_re_jk_avg),
            yerr=gv.sdev(dressing_re_jk_avg),
            label=r"$\mathrm{Re}$",
            **errorb,
        )
        ax.errorbar(
            x_values + 0.15,
            gv.mean(dressing_im_jk_avg),
            yerr=gv.sdev(dressing_im_jk_avg),
            label=r"$\mathrm{Im}$",
            **errorb,
        )
        ax.errorbar(
            x_values + 0.30,
            gv.mean(dressing_norm_jk_avg),
            yerr=gv.sdev(dressing_norm_jk_avg),
            label=r"$\mathrm{Norm}$",
            **errorb,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(momentum_label, rotation=45, ha="right")
        ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
        ax.set_ylabel(rf"${dressing_name}(p)$", **fs_p)
        ax.legend(**fs_small_p)
        plt.tight_layout()
        plt.savefig(f"artifacts/plots/qprop_dressing_{ensemble}_{dressing_name}.pdf", transparent=True)
        plt.show()

# %%
