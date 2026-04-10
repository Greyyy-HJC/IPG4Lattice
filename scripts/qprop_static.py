# %%
import os
import gvar as gv
from pyquda import init
from pyquda_utils import core, io, source, gamma
from opt_einsum import contract

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff


if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")


ensemble = "S16T16_cg_ipg"  # "S16T16", "S16T16_cg", "S16T16_cg_ipg"


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

point_quark_corr_z = []
# IPG constrains the spatially-averaged temporal links, so only the
# wall-source (spatially averaged) tdir propagator has a clean signal.
wall_quark_corr_t = []

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):

    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)

    with dirac.useGauge(gauge):
        # Z-dir: point source at the origin
        point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
        point_propag = core.invertPropagator(dirac, point_source)

        point_quark_corr_4d = core.gatherLattice(
            core.lexico(contract(
                "wtzyxijaa,ji->wtzyx",
                point_propag.data,
                I).real.get(),
            [0, 1, 2, 3, 4]),
            [0, 1, 2, 3],
        )

        # T-dir: wall source at t=0, sum over all spatial sinks
        wall_source = source.propagator(latt_info, "wall", 0)
        wall_propag = core.invertPropagator(dirac, wall_source)

        wall_quark_corr_4d = core.gatherLattice(
            core.lexico(contract(
                "wtzyxijaa,ji->wtzyx",
                wall_propag.data,
                I).real.get(),
            [0, 1, 2, 3, 4]),
            [0, 1, 2, 3],
        )

        if is_root:
            point_quark_corr_z.append(point_quark_corr_4d[0, :, 0, 0])  # t z y x
            wall_quark_corr_t.append(wall_quark_corr_4d.sum(axis=(1, 2, 3)))


# %%
if is_root:
    point_quark_corr_z = np.asarray(point_quark_corr_z)
    wall_quark_corr_t = np.asarray(wall_quark_corr_t)
    print("shape of point_quark_corr_z: ", np.shape(point_quark_corr_z))  # (N_conf, Lz)
    print("shape of wall_quark_corr_t: ", np.shape(wall_quark_corr_t))    # (N_conf, Lt)

    os.makedirs("artifacts/data", exist_ok=True)
    np.savez(
        f"artifacts/data/qprop_static_{ensemble}.npz",
        wall_quark_corr_t=wall_quark_corr_t,
        point_quark_corr_z=point_quark_corr_z,
        latt_size=latt_size,
    )
    print(f"Cached to artifacts/data/qprop_static_{ensemble}.npz")

    point_quark_corr_z_jk_avg = jk_ls_avg(jackknife(point_quark_corr_z))
    wall_quark_corr_t_jk_avg = jk_ls_avg(jackknife(wall_quark_corr_t))

    fig, ax = default_plot()

    point_meff_z = pt2_to_meff(point_quark_corr_z_jk_avg, boundary="none")
    ax.errorbar(
        np.arange(len(point_meff_z)),
        gv.mean(point_meff_z),
        yerr=gv.sdev(point_meff_z),
        label="zdir_000",
        **errorb,
    )

    wall_meff_t = pt2_to_meff(wall_quark_corr_t_jk_avg, boundary="periodic")
    ax.errorbar(
        np.arange(len(wall_meff_t)),
        gv.mean(wall_meff_t),
        yerr=gv.sdev(wall_meff_t),
        label="tdir_000",
        **errorb,
    )

    ax.legend(ncol=2, **fs_small_p)
    ax.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
    ax.set_ylim(-3, 4)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/qprop_static_meff_{ensemble}.pdf", transparent=True)
    plt.show()

# %%
