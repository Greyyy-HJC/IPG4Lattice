import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from pyquda import init
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T
from pyquda_utils.wilson_loop import wilson_loop


if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")


ensemble = "S16T16_cg_ipg"  # "S16T16", "S16T16_cg_ipg"


init([1, 1, 1, 1], resource_path=".cache")
N_conf = 50

latt_size = [16, 16, 16, 16]
r_values = np.arange(1, 8)
t_values = np.arange(1, 9)
t_fit_min = 3
t_fit_max = 5
r0_fm = 0.5

latt_info = core.LatticeInfo(latt_size)
is_root = latt_info.mpi_rank == 0

spatial_dirs = [X, Y, Z]
wilson_loops = []

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):
    if ensemble == "S16T16":
        gauge = io.readNERSCGauge(f"ensemble/S16T16/wilson_b6.{cfg}")
    elif ensemble == "S16T16_cg_ipg":
        gauge = io.readNERSCGauge(f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")

    cfg_wilson_loops = np.zeros((len(r_values), len(t_values)), dtype=np.float64)

    for r_idx, r in enumerate(r_values):
        for t_idx, t in enumerate(t_values):
            loop_dir_avg = 0.0

            for mu in spatial_dirs:
                path = [mu] * r + [T] * t + [-mu] * r + [-T] * t
                loop_link = wilson_loop(gauge, path)
                loop_lexico = core.lexico(loop_link.data.get(), [0, 1, 2, 3, 4])
                loop_trace = np.trace(loop_lexico, axis1=-2, axis2=-1)
                loop_dir_avg += np.real(loop_trace).mean() / 3.0

            cfg_wilson_loops[r_idx, t_idx] = loop_dir_avg / len(spatial_dirs)

    if is_root:
        wilson_loops.append(cfg_wilson_loops)


if is_root:
    wilson_loops = np.asarray(wilson_loops)
    wilson_loop_avg = np.mean(wilson_loops, axis=0)
    wilson_loop_jk = (np.sum(wilson_loops, axis=0)[None, :, :] - wilson_loops) / (N_conf - 1)
    v_eff_t_values = t_values[:-1]
    V_eff = np.log(wilson_loop_avg[:, :-1] / wilson_loop_avg[:, 1:])
    V_eff_jk = np.log(wilson_loop_jk[:, :, :-1] / wilson_loop_jk[:, :, 1:])

    t_fit_mask = (v_eff_t_values >= t_fit_min) & (v_eff_t_values <= t_fit_max)
    V_r = np.mean(V_eff[:, t_fit_mask], axis=1)
    V_r_jk = np.mean(V_eff_jk[:, :, t_fit_mask], axis=2)
    V_r_err = np.sqrt((N_conf - 1) * np.mean((V_r_jk - np.mean(V_r_jk, axis=0)) ** 2, axis=0))

    fit_matrix = np.column_stack([np.ones_like(r_values), 1.0 / r_values, r_values])
    fit_weight = np.diag(1.0 / V_r_err)
    fit_params = np.linalg.lstsq(fit_weight @ fit_matrix, fit_weight @ V_r, rcond=None)[0]
    V0, alpha, sigma = fit_params
    fit_params_jk = np.linalg.lstsq(fit_weight @ fit_matrix, fit_weight @ V_r_jk.T, rcond=None)[0].T
    r0_over_a = np.sqrt((1.65 + alpha) / sigma)
    r0_over_a_jk = np.sqrt((1.65 + fit_params_jk[:, 1]) / fit_params_jk[:, 2])
    a_fm = r0_fm / r0_over_a
    a_fm_jk = r0_fm / r0_over_a_jk

    fit_params_err = np.sqrt(
        (N_conf - 1) * np.mean((fit_params_jk - np.mean(fit_params_jk, axis=0)) ** 2, axis=0)
    )
    r0_over_a_err = np.sqrt((N_conf - 1) * np.mean((r0_over_a_jk - np.mean(r0_over_a_jk)) ** 2))
    a_fm_err = np.sqrt((N_conf - 1) * np.mean((a_fm_jk - np.mean(a_fm_jk)) ** 2))
    r_plot = np.linspace(np.min(r_values), np.max(r_values), 200)
    V_fit = V0 + alpha / r_plot + sigma * r_plot
    V_fit_at_r = V0 + alpha / r_values + sigma * r_values
    V_residual = V_r - V_fit_at_r

    print(f"ensemble: {ensemble}")
    print("shape of wilson_loops: ", np.shape(wilson_loops))
    print(
        "Cornell fit: "
        f"V0={V0:.8g}({fit_params_err[0]:.2g}), "
        f"alpha={alpha:.8g}({fit_params_err[1]:.2g}), "
        f"sigma={sigma:.8g}({fit_params_err[2]:.2g})"
    )
    print(f"r0/a: {r0_over_a:.8g} +/- {r0_over_a_err:.2g}")
    print(f"a [fm]: {a_fm:.8g} +/- {a_fm_err:.2g}")

    os.makedirs("artifacts/data", exist_ok=True)
    np.savez(
        f"artifacts/data/static_potential_scale_{ensemble}.npz",
        wilson_loops=wilson_loops,
        wilson_loop_avg=wilson_loop_avg,
        wilson_loop_jk=wilson_loop_jk,
        V_eff=V_eff,
        V_eff_jk=V_eff_jk,
        V_r=V_r,
        V_r_jk=V_r_jk,
        V_r_err=V_r_err,
        V_fit_at_r=V_fit_at_r,
        V_residual=V_residual,
        V0=V0,
        alpha=alpha,
        sigma=sigma,
        fit_params=fit_params,
        fit_params_jk=fit_params_jk,
        fit_params_err=fit_params_err,
        r0_over_a=r0_over_a,
        r0_over_a_jk=r0_over_a_jk,
        r0_over_a_err=r0_over_a_err,
        a_fm=a_fm,
        a_fm_jk=a_fm_jk,
        a_fm_err=a_fm_err,
        ensemble=ensemble,
        latt_size=latt_size,
        N_conf=N_conf,
        r_values=r_values,
        t_values=t_values,
        v_eff_t_values=v_eff_t_values,
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
        r0_fm=r0_fm,
    )
    print(f"Cached to artifacts/data/static_potential_scale_{ensemble}.npz")

    os.makedirs("artifacts/plots", exist_ok=True)
    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_fit.errorbar(r_values, V_r, yerr=V_r_err, fmt="o", capsize=3, label="Wilson loop data")
    ax_fit.plot(r_plot, V_fit, "-", label="Cornell fit")
    ax_fit.set_ylabel(r"$V(r)$")
    ax_fit.legend()
    ax_fit.set_title(
        rf"{ensemble}: $r_0/a={r0_over_a:.3g}\pm{r0_over_a_err:.1g}$, "
        rf"$a={a_fm:.3g}\pm{a_fm_err:.1g}\,$fm"
    )

    ax_res.axhline(0.0, color="0.3", linewidth=1)
    ax_res.errorbar(r_values, V_residual, yerr=V_r_err, fmt="o", capsize=3)
    ax_res.set_xlabel(r"$r/a$")
    ax_res.set_ylabel("resid.")

    fig.tight_layout()
    fig.savefig(f"artifacts/plots/static_potential_scale_{ensemble}.pdf", transparent=True)
    plt.close(fig)
    print(f"Saved plot to artifacts/plots/static_potential_scale_{ensemble}.pdf")
