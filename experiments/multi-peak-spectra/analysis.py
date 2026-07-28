from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from scipy import stats

plt.style.use(["science", "ieee"])

RESULTS_FOLDER = Path("../../data/results/multi-peak-spectra")

V_eth = [300, 194.33, 103.33, 0]
V_acetone = [0, 105.67, 196.67, 300]
mu_eth = 46.069
mu_acetone = 58.08
rho_eth = 0.789
rho_acetone = 0.7845
mols_eth = [V / 1000 * rho_eth / mu_eth for V in V_eth]
mols_acetone = [V / 1000 * rho_acetone / mu_acetone for V in V_acetone]

gt_molar_ratios = [mols_acetone[i] / (mols_eth[i] + mols_acetone[i]) for i in range(4)]
gt_molar_ratios = np.array(gt_molar_ratios)[[2, 1, 0, 3]]

masks = np.load(RESULTS_FOLDER / "masks.npy").transpose(2, 0, 1)
c = np.load(RESULTS_FOLDER / "c_img_fi.npy")
inserts_mask = np.any([masks[i] for i in range(4)], axis=0)
water_mask = masks[4]
c[inserts_mask, 1:] = (
    np.abs(c)[..., 1:] / np.sum(np.abs(c)[..., 1:], axis=2, keepdims=True)
)[inserts_mask]
c[water_mask] = (np.abs(c) / np.sum(np.abs(c), axis=2, keepdims=True))[water_mask]

c = c[..., 1:].astype(float)

for i, label in enumerate(["acetone", "ethanol"]):
    plt.imsave(
        RESULTS_FOLDER / f"{label}.png",
        np.abs(c[40:-40, ::-1, i]).T,
        dpi=300,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


df = pd.DataFrame(
    {
        "Ground Truth, mol/mol": [],
        "Estimation, mol/mol": [],
    }
)

for i, r in enumerate(gt_molar_ratios):
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Ground Truth, mol/mol": [r] * len(c[masks[i], 0]),
                    "Estimation, mol/mol": c[masks[i], 0],
                }
            ),
        ],
        ignore_index=True,
    )


def signed_error_stats_with_ci(errors, alpha=0.05):
    errors = np.asarray(errors, dtype=float)
    n = errors.size
    mean_err = np.mean(errors)

    if n < 2:
        return {
            "n": n,
            "mean_err": mean_err,
            "std_err": np.nan,
            "mean_ci": (np.nan, np.nan),
            "std_ci": (np.nan, np.nan),
        }

    std_err = np.std(errors, ddof=1)

    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half_width = t_crit * std_err / np.sqrt(n)
    mean_ci = (mean_err - half_width, mean_err + half_width)

    chi2_lo = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_hi = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    var = std_err**2
    std_ci = (
        np.sqrt((n - 1) * var / chi2_hi),
        np.sqrt((n - 1) * var / chi2_lo),
    )

    return {
        "n": n,
        "mean_err": mean_err,
        "std_err": std_err,
        "mean_ci": mean_ci,
        "std_ci": std_ci,
    }


# Voxel-wise signed error: e_i = gt_i - est_i
errors = (df["Ground Truth, mol/mol"] - df["Estimation, mol/mol"]).to_numpy()
metrics = signed_error_stats_with_ci(errors)

print("Acetone across all inserts (voxel-wise):")
print(f"  n={metrics['n']}")
print(
    "  Bias (mean signed error): "
    f"{metrics['mean_err']:.6f} "
    f"[95% CI: {metrics['mean_ci'][0]:.6f}, {metrics['mean_ci'][1]:.6f}] mol/mol"
)
print(
    "  Precision (SD of signed error): "
    f"{metrics['std_err']:.6f} "
    f"[95% CI: {metrics['std_ci'][0]:.6f}, {metrics['std_ci'][1]:.6f}] mol/mol"
)

cm = 1 / 2.54  # centimeters in inches
ax_width, ax_height = 6 * cm, 4 * cm  # 3.0/383*303, 2.0/383*303

left = 1.0
bottom = 1.0
right = ax_width + left
top = ax_height + bottom

fig_width = right + 0.5
fig_height = top + 0.5

f, ax = plt.subplots(figsize=(fig_width, fig_height))
f.subplots_adjust(
    left=left / fig_width,
    right=right / fig_width,
    bottom=bottom / fig_height,
    top=top / fig_height,
)
ax.plot([0, 1], [0, 1], linestyle="--", color="#FF4F4F", linewidth=0.5)
# from calibartion experiment to find the right width
ratios_calib = np.array([0.04, 0.125, 0.25, 0.5, 0.6, 0.8, 1.0])
min_dist_calib = np.min(ratios_calib[1:] - ratios_calib[:-1])
min_dist = np.min(gt_molar_ratios[1:] - gt_molar_ratios[:-1])
sns.boxplot(
    data=df,
    x="Ground Truth, mol/mol",
    y="Estimation, mol/mol",
    fill=False,
    ax=ax,
    native_scale=True,
    width=0.75 * min_dist_calib / min_dist / 2 / 21 * 23,
    gap=0.25,
    linewidth=0.5,
    fliersize=0.5,
    color="#000000",
)
ax.set_xlabel("Ground truth, mol/mol")
# remove labels and tick labels but not the ticks themselves
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.savefig(
    f"{RESULTS_FOLDER}/acetone_ratios.pdf", bbox_inches="tight", dpi=300, pad_inches=0
)
