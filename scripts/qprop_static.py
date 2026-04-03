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

# Get gamma5 matrix
I = gamma.gamma(0)


# Source positions (sources in the x-y plane at t=0, z=0)
x_src_positions = [(i * latt_size[0]) // 4 for i in range(2)]
y_src_positions = [(i * latt_size[1]) // 4 for i in range(2)]
source_positions = [[x, y, 0, 0] for x in x_src_positions for y in y_src_positions]
print("Averaging over source positions:", source_positions)

# Lists to store correlation functions
point_quark_corr_z = []
point_quark_corr_t = []

for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"../ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg":
        gauge = io.readNERSCGauge(f"../ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"../ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")
    
    # Apply smearing to gauge field
    # gauge.stoutSmear(1, 0.125, 4)

    with dirac.useGauge(gauge):
        cfg_point_quark_corr_z = []
        cfg_point_quark_corr_t = []

        for src_position in source_positions:
            src_x, src_y, src_z, src_t = src_position
            point_source = source.propagator(latt_info, "point", src_position)
            point_propag = core.invertPropagator(dirac, point_source)

            # Gather the point-source correlator in [t, z, y, x] order.
            point_quark_corr_4d = core.gatherLattice(
                core.lexico(contract(
                    "wtzyxijaa,ji->wtzyx",
                    point_propag.data,
                    I).real.get(),
                [0, 1, 2, 3, 4]),
                [0, 1, 2, 3],
            )

            cfg_point_quark_corr_z.append(point_quark_corr_4d[src_t, :, src_y, src_x]) # t z y x
            cfg_point_quark_corr_t.append(point_quark_corr_4d[:, src_z, src_y, src_x]) # t z y x

        point_quark_corr_z.append(np.mean(cfg_point_quark_corr_z, axis=0)) # avg over source positions
        point_quark_corr_t.append(np.mean(cfg_point_quark_corr_t, axis=0))


# %%
if latt_info.mpi_rank == 0:
    print("shape of point_quark_corr_z: ", np.shape(point_quark_corr_z)) # Should be (N_sample, Lz)
    print("shape of point_quark_corr_t: ", np.shape(point_quark_corr_t)) # Should be (N_sample, Lt)

    point_quark_corr_z_jk = jackknife(point_quark_corr_z)
    point_quark_corr_z_jk_avg = jk_ls_avg(point_quark_corr_z_jk)

    point_quark_corr_t_jk = jackknife(point_quark_corr_t)
    point_quark_corr_t_jk_avg = jk_ls_avg(point_quark_corr_t_jk)

    fig, ax = default_plot()
    point_meff_z = pt2_to_meff(point_quark_corr_z_jk_avg, boundary="none")

    ax.errorbar(
        np.arange(len(point_meff_z)),
        gv.mean(point_meff_z),
        yerr=gv.sdev(point_meff_z),
        label="zdir_000",
        **errorb,
    )

    point_meff_t = pt2_to_meff(point_quark_corr_t_jk_avg, boundary="none")

    ax.errorbar(
        np.arange(len(point_meff_t)),
        gv.mean(point_meff_t),
        yerr=gv.sdev(point_meff_t),
        label="tdir_000",
        **errorb,
    )

    ax.legend(ncol=2, **fs_small_p)
    ax.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
    ax.set_ylim(-3, 4)
    plt.tight_layout()
    plt.savefig(f"../artifacts/plots/qprop_static_meff_{ensemble}.pdf", transparent=True)
    plt.show()

# %%
