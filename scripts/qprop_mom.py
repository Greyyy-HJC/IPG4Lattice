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

I = gamma.gamma(0)
gX = gamma.gamma(1)
gY = gamma.gamma(2)
gZ = gamma.gamma(4)
gT = gamma.gamma(8)

# Spatial momentum phases (3-momenta); at p=(0,0,0) the effective mass
# agrees with qprop_static's tdir (absolute value differs by L^3).
momentum_list = [[0, 0, 0], [1, 1, 1], [0, 0, 2], [2, 2, 2]]
momentum_label = ["000", "111", "002", "222"]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list)

gamma_ops = [("I", I), ("gX", gX), ("gY", gY), ("gZ", gZ), ("gT", gT)]
# Use momentum wall sources: one inversion per momentum, L^3 better statistics
# than a single point source. Especially important for the IPG ensemble where
# IPG constrains spatially-averaged links — the wall-source signal is much cleaner.
wall_quark_corr_t_by_gamma = {gamma_name: [] for gamma_name, _ in gamma_ops}

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):
    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)

    with dirac.useGauge(gauge):
        # One wall-source inversion per momentum.
        # Source phase: exp(-ip·x_src), sink phase: exp(+ip·x_snk).
        # By translation invariance this equals L^3 * C_point(t, p), with
        # sqrt(L^3) better SNR due to averaging over all spatial source sites.
        cfg_corr = {gamma_name: [] for gamma_name, _ in gamma_ops}

        for p_phase in momentum_phases:
            wall_source = source.propagator(latt_info, "wall", 0,
                                            source_phase=np.conj(p_phase))
            wall_propag = core.invertPropagator(dirac, wall_source)

            for gamma_name, gamma_matrix in gamma_ops:
                # p_phase[None] adds a dummy p-axis so gatherLattice (which
                # needs at least a 2D input for axis-1 time gathering) works.
                corr_t = core.gatherLattice(
                    contract(
                        "pwtzyx,wtzyxijaa,ji->pt",
                        p_phase[None],
                        wall_propag.data,
                        gamma_matrix,
                    ).get(),
                    [1, -1, -1, -1],
                )[0]  # remove dummy p-axis
                if is_root:
                    cfg_corr[gamma_name].append(corr_t)

        if is_root:
            for gamma_name in cfg_corr:
                # stack to shape (n_mom, Lt) then accumulate over configs
                wall_quark_corr_t_by_gamma[gamma_name].append(
                    np.stack(cfg_corr[gamma_name])
                )

if is_root:
    for gamma_name in [name for name, _ in gamma_ops]:
        point_quark_corr_t = np.asarray(wall_quark_corr_t_by_gamma[gamma_name])
        print("max |Im point_quark_corr_t|: ", np.max(np.abs(np.imag(point_quark_corr_t))))
        point_quark_corr_t = np.real(point_quark_corr_t)
        print("shape of point_quark_corr_t: ", np.shape(point_quark_corr_t))  # (N_conf, n_mom, Lt)

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
