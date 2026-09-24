from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns

plt.style.use(["science", "ieee"])

RESULTS_FOLDER = Path("../../data/results/composition-dependent-spectra")

gt_molar_ratios = np.array([0.0577005, 0.140387, 0.268686, 0.494995])

nh_water = 2
nh_acetone = 6


def ratio(c):
    r = np.abs(c[..., 0] / nh_water) / (
        np.abs(c[..., 0] / nh_water) + np.abs(c[..., 1] / nh_acetone)
    )
    return np.stack([r, 1 - r], axis=-1)


masks = np.load(RESULTS_FOLDER / "masks.npy").transpose(2, 0, 1)
phantom_mask = np.any([masks[i] for i in range(5)], axis=0)
c_static = np.load(RESULTS_FOLDER / "c_img_fi.npy")
c_static = ratio(c_static) * phantom_mask[..., np.newaxis]
c_linear_approx_ratio_dependent = np.load(
    RESULTS_FOLDER / "c_img_linear_approx_ratio_dependent.npy"
)
c_linear_approx_ratio_dependent = (
    ratio(c_linear_approx_ratio_dependent) * phantom_mask[..., np.newaxis]
)


for method, c in zip(
    ["static", "linear_approx_ratio_dependent"],
    [c_static, c_linear_approx_ratio_dependent],
):
    for i, label in enumerate(["water", "acetone"]):
        plt.imsave(
            RESULTS_FOLDER / f"{method}_{label}.png",
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
                        "Estimation, mol/mol": c[masks[i], 1],
                    }
                ),
            ],
            ignore_index=True,
        )

    print(f"Method {method}")
    print("\tAcetone by insert (single measurement):")
    biases = []
    for i, ground_truth in enumerate(gt_molar_ratios):
        estimates = c[masks[i], 1]
        mean_estimate = np.mean(estimates)
        spatial_variation = np.std(estimates, ddof=1)
        biases.append(ground_truth - mean_estimate)
        print(
            f"\t  Insert {i + 1}: bias={ground_truth - mean_estimate:.6f}, "
            f"spatial variation={spatial_variation:.6f} mol/mol"
        )
    print(f"\tMean bias: {np.mean(biases):.6f}")
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
    ax.plot([0, 0.6], [0, 0.6], linestyle="--", color="#FF4F4F", linewidth=0.5)
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
        width=0.75 * min_dist_calib / min_dist / 2 / 21 * 23 * 0.6,
        gap=0.25,
        linewidth=0.5,
        fliersize=0.5,
        color="#000000",
    )
    ax.set_xlabel("Ground truth, mol/mol")
    # remove labels and tick labels but not the ticks themselves
    plt.xlim(-0.05, 0.65)
    plt.ylim(-0.05, 0.65)
    if method == "static":
        # remove xlabel
        ax.set_xlabel("")
    # ticks from 0 to 0.6
    ax.set_xticks(np.arange(0, 0.65, 0.12))
    ax.set_yticks(np.arange(0, 0.65, 0.12))
    # set minor ticks only between the major ticks
    ax.set_xticks(np.arange(0, 0.605, 0.03), minor=True)
    ax.set_yticks(np.arange(0, 0.605, 0.03), minor=True)
    plt.savefig(
        f"{RESULTS_FOLDER}/acetone_ratios_{method}.pdf",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0,
    )
