import os
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *


ensemble = "S16T16_cg_ipg"  # "S16T16", "S16T16_cg", "S16T16_cg_ipg"
cache_path = f"artifacts/data/qprop_mom_{ensemble}.npz"

if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"Cache not found: {cache_path}. Run scripts/qprop_mom.py first."
    )

cache = np.load(cache_path)
momentum_label = np.asarray(cache["momentum_label"], dtype=str)

meta_keys = {"momentum_list", "momentum_label", "latt_size"}
gamma_keys = [key for key in cache.files if key not in meta_keys]

os.makedirs("artifacts/plots", exist_ok=True)

for gamma_name in gamma_keys:
    # Shape: (N_conf, n_mom, Lt)
    corr = np.asarray(cache[gamma_name])
    print(f"{gamma_name}: shape = {corr.shape}")

    corr_re_jk_avg = jk_ls_avg(jackknife(np.real(corr)))
    corr_im_jk_avg = jk_ls_avg(jackknife(np.imag(corr)))
    corr_norm_jk_avg = jk_ls_avg(jackknife(np.abs(corr)))

    x_t = np.arange(corr.shape[-1])

    fig_reim, ax_reim = default_plot()
    fig_norm, ax_norm = default_plot()

    for idx, label in enumerate(momentum_label):
        ax_reim.errorbar(
            x_t,
            gv.mean(corr_re_jk_avg[idx]),
            yerr=gv.sdev(corr_re_jk_avg[idx]),
            label=f"Re tdir_{label}",
            **errorb,
        )
        ax_reim.errorbar(
            x_t + 0.12,
            gv.mean(corr_im_jk_avg[idx]),
            yerr=gv.sdev(corr_im_jk_avg[idx]),
            label=f"Im tdir_{label}",
            **errorb,
        )
        ax_norm.errorbar(
            x_t,
            gv.mean(corr_norm_jk_avg[idx]),
            yerr=gv.sdev(corr_norm_jk_avg[idx]),
            label=f"Norm tdir_{label}",
            **errorb,
        )

    ax_reim.legend(ncol=2, **fs_small_p)
    ax_reim.set_xlabel(r"$t$", **fs_p)
    ax_reim.set_ylabel(rf"$C_{{\Gamma={gamma_name}}}(t,\vec{{p}})$", **fs_p)
    fig_reim.tight_layout()
    # fig_reim.savefig(
    #     f"artifacts/plots/qprop_mom_tdir_c2pt_reim_{ensemble}_{gamma_name}.pdf",
    #     transparent=True,
    # )
    plt.show()

    ax_norm.legend(ncol=2, **fs_small_p)
    ax_norm.set_xlabel(r"$t$", **fs_p)
    ax_norm.set_ylabel(rf"$|C_{{\Gamma={gamma_name}}}(t,\vec{{p}})|$", **fs_p)
    fig_norm.tight_layout()
    # fig_norm.savefig(
    #     f"artifacts/plots/qprop_mom_tdir_c2pt_norm_{ensemble}_{gamma_name}.pdf",
    #     transparent=True,
    # )
    plt.show()
