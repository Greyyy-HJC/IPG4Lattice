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
is_root = latt_info.mpi_rank == 0

# Get gamma5 matrix
I = gamma.gamma(0)
gX = gamma.gamma(1)
gY = gamma.gamma(2)
gZ = gamma.gamma(4)
gT = gamma.gamma(8)

# Momentum phases
momentum_list = [[0, 0, 0], [1, 1, 1], [0, 0, 2], [2, 2, 2]]
momentum_label = ["000", "111", "002", "222"]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list)

# Source positions
source_positions = [[0, 0, 0, 0]]
if is_root:
    print("Averaging over source positions:", source_positions)

# Lists to store correlation functions
for gamma_name, gamma_matrix in zip(["I", "gX", "gY", "gZ", "gT"], [I, gX, gY, gZ, gT]):
    point_quark_corr_z = []
    point_quark_corr_t = []

    for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):
        
        if ensemble == "S16T16":
            gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
        elif ensemble == "S16T16_cg":
            gauge = io.readNERSCGauge(f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
        elif ensemble == "S16T16_cg_ipg":
            gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")
        
        # Apply smearing to gauge field
        # gauge.stoutSmear(1, 0.125, 4)

        with dirac.useGauge(gauge):
            cfg_point_quark_corr_t = []

            for src_position in source_positions:
                src_x, src_y, src_z, src_t = src_position
                point_source = source.propagator(latt_info, "point", src_position)
                point_propag = core.invertPropagator(dirac, point_source)

                # Gather the point-source correlator in [t, z, y, x] order.
                point_quark_corr_4d = core.gatherLattice(
                    contract(
                        "pwtzyx,wtzyxijaa,ji->pt",
                        momentum_phases,
                        point_propag.data,
                        gamma_matrix).get(),
                    [1, -1, -1, -1],
                )

                if is_root:
                    cfg_point_quark_corr_t.append(point_quark_corr_4d)

            if is_root:
                point_quark_corr_t.append(np.mean(cfg_point_quark_corr_t, axis=0))


    if is_root:
        point_quark_corr_t = np.asarray(point_quark_corr_t)
        print("max |Im point_quark_corr_t|: ", np.max(np.abs(np.imag(point_quark_corr_t))))
        point_quark_corr_t = np.real(point_quark_corr_t)
        print("shape of point_quark_corr_t: ", np.shape(point_quark_corr_t)) # Should be (N_sample, len(momentum_list), Lt)

        point_quark_corr_t_jk = jackknife(point_quark_corr_t)
        point_quark_corr_t_jk_avg = jk_ls_avg(point_quark_corr_t_jk)

        fig, ax = default_plot()
        for idx, label in enumerate(momentum_label):

            point_meff_t = pt2_to_meff(point_quark_corr_t_jk_avg[idx], boundary="none")

            ax.errorbar(
                np.arange(len(point_meff_t)),
                gv.mean(point_meff_t),
                yerr=gv.sdev(point_meff_t),
                label="tdir_" + label,
                **errorb,
            )

        ax.legend(ncol=2, **fs_small_p)
        ax.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
        ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
        ax.set_ylim(-2, 4)
        plt.tight_layout()
        plt.savefig(f"artifacts/plots/qprop_mom_tdir_meff_{ensemble}_{gamma_name}.pdf", transparent=True)
        plt.show()

# %%
