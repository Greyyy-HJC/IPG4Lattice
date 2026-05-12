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

N_src = 4
rng = np.random.RandomState(42)
if is_root:
    print(f"Using {N_src} random source positions per config, temporal momenta n4 = {temporal_momenta}")

# Accumulate per-gamma Green's functions across configurations
point_quark_greens_by_gamma = {name: [] for name, _ in gamma_ops}
point_quark_greens_by_gamma["pDotg"] = []

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
            cfg_mean_by_gamma["pDotg"] = (
                momentum_array[:, 0] * cfg_mean_by_gamma["gX"]
                + momentum_array[:, 1] * cfg_mean_by_gamma["gY"]
                + momentum_array[:, 2] * cfg_mean_by_gamma["gZ"]
            )
            for gamma_name in cfg_mean_by_gamma:
                point_quark_greens_by_gamma[gamma_name].append(cfg_mean_by_gamma[gamma_name])


# %%
if is_root:
    # Cache complex Green's functions for all gammas
    cache_dict = dict(
        momentum_list=np.asarray(momentum_list),
        momentum_label=np.asarray(momentum_label),
        latt_size=np.asarray(latt_size),
    )
    for gamma_name in point_quark_greens_by_gamma:
        cache_dict[gamma_name] = np.asarray(point_quark_greens_by_gamma[gamma_name])

    os.makedirs("artifacts/data", exist_ok=True)
    np.savez(f"artifacts/data/qprop_greens_{ensemble}.npz", **cache_dict)
    print(f"Cached to artifacts/data/qprop_greens_{ensemble}.npz")

    # Plot each gamma separately
    for gamma_name in point_quark_greens_by_gamma:
        greens = np.asarray(point_quark_greens_by_gamma[gamma_name])
        greens_re = np.real(greens)
        greens_im = np.imag(greens)
        greens_norm = np.abs(greens)

        greens_re_jk_avg = jk_ls_avg(jackknife(greens_re))
        greens_im_jk_avg = jk_ls_avg(jackknife(greens_im))
        greens_norm_jk_avg = jk_ls_avg(jackknife(greens_norm))

        x_values = np.arange(len(momentum_label))

        fig, ax = default_plot()
        ax.errorbar(
            x_values,
            gv.mean(greens_re_jk_avg),
            yerr=gv.sdev(greens_re_jk_avg),
            label=r"$\mathrm{Re}$",
            **errorb,
        )
        ax.errorbar(
            x_values + 0.15,
            gv.mean(greens_im_jk_avg),
            yerr=gv.sdev(greens_im_jk_avg),
            label=r"$\mathrm{Im}$",
            **errorb,
        )
        ax.errorbar(
            x_values + 0.30,
            gv.mean(greens_norm_jk_avg),
            yerr=gv.sdev(greens_norm_jk_avg),
            label=r"$\mathrm{Norm}$",
            **errorb,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(momentum_label, rotation=45, ha="right")
        ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
        ax.set_ylabel(rf"$G_{{\Gamma={gamma_name}}}(p)$", **fs_p)
        ax.legend(**fs_small_p)
        plt.tight_layout()
        plt.savefig(f"artifacts/plots/qprop_greens_{ensemble}_{gamma_name}.pdf", transparent=True)
        plt.show()

# %%
