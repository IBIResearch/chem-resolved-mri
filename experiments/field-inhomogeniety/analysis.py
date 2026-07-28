from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from scipy import stats

plt.style.use(["science", "ieee"])


RESULTS_FOLDER = Path("../../data/results/field-inhomogeniety")

masks = np.load(RESULTS_FOLDER / "masks.npy").transpose(2, 0, 1)
masks = masks[:, 40:-40]
nh_water = 2
nh_acetone = 6

for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing"]:
    c = np.load(RESULTS_FOLDER / f"{name}.npy")
    c = c[40:-40]
    c[..., 0] /= nh_water
    c[..., 1] /= nh_acetone
    c = (
        np.abs(c)
        / np.sum(np.abs(c), axis=2, keepdims=True)
        * np.any([masks[i] for i in range(5)], axis=0)[..., None]
    )

    plt.imsave(
        RESULTS_FOLDER / f"{name}_abs_0.png",
        np.abs(c[:, ::-1, 0]).T,
        dpi=300,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    plt.imsave(
        RESULTS_FOLDER / f"{name}_abs_1.png",
        np.abs(c[:, ::-1, 1]).T,
        dpi=300,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

cm = 1 / 2.54  # centimeters in inches
ax_width, ax_height = 6 * cm, 4 * cm  # 3.0/383*303, 2.0/383*303

left = 1.0
bottom = 1.0
right = ax_width + left
top = ax_height + bottom

fig_width = right + 0.5
fig_height = top + 0.5

colors = ["#7200FE", "#143BFF", "#FFDE36", "#FF7E15"]
for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing"]:
    f, ax = plt.subplots(figsize=(fig_width, fig_height))
    f.subplots_adjust(
        left=left / fig_width,
        right=right / fig_width,
        bottom=bottom / fig_height,
        top=top / fig_height,
    )
    for i in range(4):
        mask = masks[i]
        c = np.load(RESULTS_FOLDER / f"{name}.npy")
        c = c[40:-40]
        c[..., 0] /= nh_water
        c[..., 1] /= nh_acetone
        c = np.abs(c) / np.sum(np.abs(c), axis=2, keepdims=True)
        channel = 1
        values = c[..., channel][mask]

        ax.hist(values, bins=20, color=colors[i], alpha=0.7)
    ax.axvline(0.25, color="#FF4F4F", linestyle="--", linewidth=1, label="Ground truth")
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    # ax.grid(visible=True, axis='both', linestyle='-', linewidth=0.5, color='gray')
    if name == "c_img_fi_compressed_sensing":
        ax.set_xlabel("Estimated acetone molar ratio, mol/mol")
    plt.savefig(
        RESULTS_FOLDER / f"{name}_hist.pdf", bbox_inches="tight", dpi=300, pad_inches=0
    )
    plt.close()

mean_acetone_ratios = {}
gt_ratio = 0.25


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


for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing"]:
    mean_acetone_ratios[name] = []
    c = np.load(RESULTS_FOLDER / f"{name}.npy")
    c = c[40:-40]
    c[..., 0] /= nh_water
    c[..., 1] /= nh_acetone
    c = np.abs(c) / np.sum(np.abs(c), axis=2, keepdims=True)
    acetone_ratio = c[..., 1]

    acetone_ratio_values = acetone_ratio[np.any([masks[i] for i in range(4)], axis=0)]
    # Voxel-wise signed error: e_i = gt_i - est_i
    errors = gt_ratio - acetone_ratio_values
    metrics = signed_error_stats_with_ci(errors)

    for i in range(4):
        mean_acetone_ratios[name].append(np.mean(acetone_ratio[masks[i]]))

    print(f"{name}:")
    print(f"\t n={metrics['n']}")
    print(
        "\t Bias (mean signed error): "
        f"{metrics['mean_err']:.4f} "
        f"[95% CI: {metrics['mean_ci'][0]:.4f}, {metrics['mean_ci'][1]:.4f}]"
    )
    print(
        "\t Precision (SD of signed error): "
        f"{metrics['std_err']:.4f} "
        f"[95% CI: {metrics['std_ci'][0]:.4f}, {metrics['std_ci'][1]:.4f}]"
    )
