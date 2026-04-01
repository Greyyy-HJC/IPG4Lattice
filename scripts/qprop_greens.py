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
gX = gamma.gamma(1)
gY = gamma.gamma(2)
gZ = gamma.gamma(4)
gT = gamma.gamma(8)

# Momentum phases
spatial_momenta = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 2]]
temporal_momenta = [0, 1, 2]
momentum_list = [[px, py, pz, pt] for px, py, pz in spatial_momenta for pt in temporal_momenta]
momentum_label = [f"({px},{py},{pz},{pt})" for px, py, pz, pt in momentum_list]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list)

# Source positions
source_positions = [[0, 0, 0, 0]]
print("Averaging over source positions:", source_positions)

# Lists to store correlation functions
for gamma_name, gamma_matrix in zip(["I"], [I]):
    point_quark_greens = []

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
            cfg_point_quark_greens = []

            for src_position in source_positions:
                src_x, src_y, src_z, src_t = src_position
                point_source = source.propagator(latt_info, "point", src_position)
                point_propag = core.invertPropagator(dirac, point_source)

                point_quark_green = contract(
                    "pwtzyx,wtzyxijaa,ji->p",
                    momentum_phases,
                    point_propag.data,
                    gamma_matrix,
                ).get()

                cfg_point_quark_greens.append(point_quark_green)

            point_quark_greens.append(np.mean(cfg_point_quark_greens, axis=0))


    if latt_info.mpi_rank == 0:
        point_quark_greens = np.asarray(point_quark_greens)
        print("max |Im point_quark_greens|: ", np.max(np.abs(np.imag(point_quark_greens))))
        point_quark_greens = np.real(point_quark_greens)
        print("shape of point_quark_greens: ", np.shape(point_quark_greens)) # Should be (N_sample, len(momentum_list))

        point_quark_greens_jk = jackknife(point_quark_greens)
        point_quark_greens_jk_avg = jk_ls_avg(point_quark_greens_jk)

        fig, ax = default_plot()
        x_values = np.arange(len(momentum_label))
        ax.errorbar(
            x_values,
            gv.mean(point_quark_greens_jk_avg),
            yerr=gv.sdev(point_quark_greens_jk_avg),
            **errorb,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(momentum_label, rotation=45, ha="right")
        ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
        ax.set_ylabel(r"$\mathrm{Re}\, G(p)$", **fs_p)
        plt.tight_layout()
        plt.savefig(f"../artifacts/plots/qprop_greens_{ensemble}_{gamma_name}.pdf", transparent=True)
        plt.show()

# %%
