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
mass = -0.038888  # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None  # [[4, 4, 4, 4], [2, 2, 2, 8]]

latt_size = [16, 16, 16, 16]
latt_info = core.LatticeInfo(latt_size, -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)
is_root = latt_info.mpi_rank == 0

I = gamma.gamma(0)

point_quark_corr_z = []
point_quark_corr_t = []

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):

    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)

    with dirac.useGauge(gauge):
        point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
        point_propag = core.invertPropagator(dirac, point_source)

        point_quark_corr_4d = core.gatherLattice(
            core.lexico(
                contract(
                    "wtzyxijaa,ji->wtzyx",
                    point_propag.data,
                    I,
                ).real.get(),
                [0, 1, 2, 3, 4],
            ),
            [0, 1, 2, 3],
        )

        if is_root:
            point_quark_corr_z.append(point_quark_corr_4d[0, :, 0, 0])  # t z y x
            # Point source at the origin, summed over spatial sinks for each t.
            point_quark_corr_t.append(point_quark_corr_4d.sum(axis=(1, 2, 3)))
            # point_quark_corr_t.append(point_quark_corr_4d[:, 0, 0, 0])


if is_root:
    point_quark_corr_z = np.asarray(point_quark_corr_z)
    point_quark_corr_t = np.asarray(point_quark_corr_t)
    point_quark_ft_t = np.fft.fft(point_quark_corr_t, axis=1)
    point_quark_ft_t_zero = np.real(point_quark_ft_t[:, 0])

    print("shape of point_quark_corr_z: ", np.shape(point_quark_corr_z))  # (N_conf, Lz)
    print("shape of point_quark_corr_t: ", np.shape(point_quark_corr_t))  # (N_conf, Lt)
    print("shape of point_quark_ft_t: ", np.shape(point_quark_ft_t))      # (N_conf, Lt)

    os.makedirs("artifacts/data", exist_ok=True)
    np.savez(
        f"artifacts/data/qprop_static_pt_{ensemble}.npz",
        point_quark_corr_t=point_quark_corr_t,
        point_quark_corr_z=point_quark_corr_z,
        point_quark_ft_t=point_quark_ft_t,
        point_quark_ft_t_zero=point_quark_ft_t_zero,
        latt_size=latt_size,
    )
    print(f"Cached to artifacts/data/qprop_static_pt_{ensemble}.npz")

    point_quark_corr_z_jk_avg = jk_ls_avg(jackknife(point_quark_corr_z))
    point_quark_corr_t_jk_avg = jk_ls_avg(jackknife(point_quark_corr_t))
    point_quark_ft_t_zero_jk_avg = jk_ls_avg(jackknife(point_quark_ft_t_zero))

    greens_cache_path = f"artifacts/data/qprop_greens_{ensemble}.npz"
    if os.path.exists(greens_cache_path):
        greens_cache = np.load(greens_cache_path)
        momentum_labels = np.asarray(greens_cache["momentum_label"], dtype=str)
        zero_mode_idx = np.where(momentum_labels == "(0,0,0,0)")[0]
        if len(zero_mode_idx) > 0 and "I" in greens_cache:
            greens_zero_mode = np.real(greens_cache["I"][:, zero_mode_idx[0]])
            greens_zero_mode_jk_avg = jk_ls_avg(jackknife(greens_zero_mode))
            zero_mode_diff_jk_avg = jk_ls_avg(jackknife(point_quark_ft_t_zero - greens_zero_mode))
            print(
                "FT_t point-source zero mode:",
                point_quark_ft_t_zero_jk_avg,
            )
            print(
                "qprop_greens I(p=0,0,0,0):",
                greens_zero_mode_jk_avg,
            )
            print(
                "difference [FT_t - greens]:",
                zero_mode_diff_jk_avg,
            )

    fig, ax = default_plot()

    point_meff_z = pt2_to_meff(point_quark_corr_z_jk_avg, boundary="none")
    ax.errorbar(
        np.arange(len(point_meff_z)),
        gv.mean(point_meff_z),
        yerr=gv.sdev(point_meff_z),
        label="zdir_000",
        **errorb,
    )

    point_meff_t = pt2_to_meff(point_quark_corr_t_jk_avg, boundary="periodic")
    ax.errorbar(
        np.arange(len(point_meff_t)),
        gv.mean(point_meff_t),
        yerr=gv.sdev(point_meff_t),
        label="tdir_000_point",
        **errorb,
    )

    # The p_t = 0 temporal Fourier mode is a scalar, so plot it as a band on a
    # second y-axis while keeping the effective mass scale readable.
    ax_ft = ax.twinx()
    ft_zero_mean = gv.mean(point_quark_ft_t_zero_jk_avg)
    ft_zero_sdev = gv.sdev(point_quark_ft_t_zero_jk_avg)
    x_band = np.arange(len(point_meff_t))
    ax_ft.fill_between(
        x_band,
        np.full_like(x_band, ft_zero_mean - ft_zero_sdev, dtype=float),
        np.full_like(x_band, ft_zero_mean + ft_zero_sdev, dtype=float),
        color="tab:red",
        alpha=0.2,
        label=r"$\tilde{C}(p_t=0)$",
    )
    ax_ft.set_ylabel(r"$\tilde{C}(p_t=0)$", **fs_p)

    handles, labels = ax.get_legend_handles_labels()
    handles_ft, labels_ft = ax_ft.get_legend_handles_labels()
    ax.legend(handles + handles_ft, labels + labels_ft, ncol=2, **fs_small_p)
    ax.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
    ax.set_ylim(-3, 4)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/qprop_static_pt_meff_{ensemble}.pdf", transparent=True)
    plt.show()
