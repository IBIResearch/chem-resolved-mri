from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa

plt.style.use(["science", "ieee"])


def nh2molar(ratio):
    return ratio / (3 - 2 * ratio)


RESULTS_FOLDER = Path("../../data/results/field-inhomogeniety")

masks = np.load(RESULTS_FOLDER / "masks.npy").transpose(2, 0, 1)
masks = masks[:, 40:-40]
nh_water = 2
nh_acetone = 6

for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing", "varpro"]:
    if name != "varpro":
        c = np.load(RESULTS_FOLDER / f"{name}.npy")
        c = c[40:-40]
        c[..., 0] /= nh_water
        c[..., 1] /= nh_acetone
        c = (
            np.abs(c)
            / np.sum(np.abs(c), axis=2, keepdims=True)
            * np.any([masks[i] for i in range(5)], axis=0)[..., None]
        )
    else:
        fat_fraction = np.load(RESULTS_FOLDER / "varpro_fat_fraction_map.npy")
        fat_fraction = nh2molar(fat_fraction)
        c = np.stack([1 - fat_fraction, fat_fraction], axis=-1)
        c = c[40:-40] * np.any([masks[i] for i in range(4)], axis=0)[..., None]

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
for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing", "varpro"]:
    f, ax = plt.subplots(figsize=(fig_width, fig_height))
    f.subplots_adjust(
        left=left / fig_width,
        right=right / fig_width,
        bottom=bottom / fig_height,
        top=top / fig_height,
    )
    for i in range(4):
        mask = masks[i]
        if name == "varpro":
            fat_fraction = np.load(RESULTS_FOLDER / "varpro_fat_fraction_map.npy")
            fat_fraction = nh2molar(fat_fraction)
            c = np.stack([1 - fat_fraction, fat_fraction], axis=-1)
            c = c[40:-40] * np.any([masks[i] for i in range(4)], axis=0)[..., None]
        else:
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


for name in ["c_img", "c_img_fi", "c_img_fi_compressed_sensing", "varpro"]:
    mean_acetone_ratios[name] = []
    if name == "varpro":
        fat_fraction = np.load(RESULTS_FOLDER / "varpro_fat_fraction_map.npy")
        fat_fraction = nh2molar(fat_fraction)
        c = np.stack([1 - fat_fraction, fat_fraction], axis=-1)
        c = c[40:-40] * np.any([masks[i] for i in range(4)], axis=0)[..., None]
    else:
        c = np.load(RESULTS_FOLDER / f"{name}.npy")
        c = c[40:-40]
        c[..., 0] /= nh_water
        c[..., 1] /= nh_acetone
        c = np.abs(c) / np.sum(np.abs(c), axis=2, keepdims=True)
    acetone_ratio = c[..., 1]

    biases = []
    for i in range(4):
        estimates = acetone_ratio[masks[i]]
        mean_estimate = np.mean(estimates)
        spatial_variation = np.std(estimates, ddof=1)
        biases.append(gt_ratio - mean_estimate)
        mean_acetone_ratios[name].append(mean_estimate)
        print(
            f"{name}, insert {i + 1}: bias={gt_ratio - mean_estimate:.4f}, "
            f"spatial variation={spatial_variation:.4f}"
        )
    print(f"{name} mean bias: {np.mean(biases):.4f}")


# save CS mask
cs_mask = np.load(RESULTS_FOLDER / "cs_mask.npy")
cs_mask = cs_mask[40:-40]
plt.imsave(
    RESULTS_FOLDER / "cs_mask.png",
    cs_mask[:, ::-1].T,
    dpi=300,
    cmap="gray",
    vmin=0,
    vmax=1,
)
