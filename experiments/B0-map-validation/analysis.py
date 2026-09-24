from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes

RESULTS_FOLDER = Path("../../data/results/B0-map-validation")
estimated_slopes = (
    np.load(RESULTS_FOLDER / "water_phases_slopes_water_cyliner_extrapolated.npy")
    * 1000.0
    / (2 * np.pi)
)
slopes_outer_cylinder = (
    np.load(RESULTS_FOLDER / "water_phases_slopes_water_cyliner.npy")
    * 1000.0
    / (2 * np.pi)
)
masks = np.load(RESULTS_FOLDER / "masks.npy")

measured_slopes = np.zeros((2,) + masks.shape[:2])
comparison_rows = []
for echo_direction in range(2):
    # Julia's boolean indexing follows column-major order.  Transposing both
    # the target and mask makes NumPy's C-order boolean indexing use that same
    # order while preserving the original image orientation.
    measured_slopes[echo_direction].T[masks[..., 4].T] = slopes_outer_cylinder[
        :, 0, echo_direction
    ]
    for i in range(4):
        slopes_tube = (
            np.load(RESULTS_FOLDER / f"water_phases_slopes_tube_{i+1}.npy")[
                :, 0, echo_direction
            ]
            * 1000.0
            / (2 * np.pi)
        )
        estimation = estimated_slopes[masks[..., i], 0, echo_direction]
        comparison_rows.extend(
            {
                "Tube": i + 1,
                "Echo direction": ["odd", "even"][echo_direction],
                "Measured, Hz": measured,
                "Estimated, Hz": estimated,
            }
            for measured, estimated in zip(slopes_tube, estimation)
        )
        measured_slopes[echo_direction].T[masks[..., i].T] = slopes_tube


for echo_direction in ["odd", "even"]:
    for tube in range(1, 5):
        rows_for_insert = [
            row
            for row in comparison_rows
            if row["Echo direction"] == echo_direction and row["Tube"] == tube
        ]
        measured = np.asarray([row["Measured, Hz"] for row in rows_for_insert])
        estimated = np.asarray([row["Estimated, Hz"] for row in rows_for_insert])
        print(
            f"{echo_direction} echoes, insert {tube}: "
            f"bias={np.mean(measured) - np.mean(estimated):.6f} Hz, "
            f"spatial variation={np.std(estimated, ddof=1):.6f} Hz"
        )

cmap_freq_offset = plt.get_cmap("gray").copy()
cmap_freq_offset.set_bad(alpha=0)


cmap_difference = plt.get_cmap("coolwarm").copy()
cmap_difference.set_bad(alpha=0)


f, axes = plt.subplots(3, 3, figsize=(12 * 2, 8 * 2))
for ax in axes.flatten():
    ax.set_axis_off()

slopes_range = (5, 18)
diff_range = (-5.0, 5.0)
total_mask = np.logical_or.reduce(masks, axis=2)
for echo_direction in range(2):
    meas = measured_slopes[echo_direction][40:-40].T[:, ::-1]
    meas[~total_mask[40:-40].T[:, ::-1]] = np.nan
    measured_image = axes[echo_direction, 0].imshow(
        meas, cmap=cmap_freq_offset, vmin=slopes_range[0], vmax=slopes_range[1]
    )

    est = estimated_slopes[..., 0, echo_direction]
    est[~binary_fill_holes(masks[..., 4])] = np.nan
    est = est[40:-40].T[:, ::-1]
    estimated_image = axes[echo_direction, 1].imshow(
        est, cmap=cmap_freq_offset, vmin=slopes_range[0], vmax=slopes_range[1]
    )
    diff = measured_slopes[echo_direction] - estimated_slopes[..., 0, echo_direction]
    diff[~total_mask] = np.nan
    diff = diff[40:-40].T[:, ::-1]
    difference_image = axes[echo_direction, 2].imshow(
        diff, cmap=cmap_difference, vmin=diff_range[0], vmax=diff_range[1]
    )

    echo_direction_name = ["odd", "even"][echo_direction]
    plt.imsave(
        RESULTS_FOLDER / f"measured_{echo_direction_name}.png",
        meas,
        dpi=300,
        cmap=cmap_freq_offset,
        vmin=slopes_range[0],
        vmax=slopes_range[1],
    )
    plt.imsave(
        RESULTS_FOLDER / f"estimated_{echo_direction_name}.png",
        est,
        dpi=300,
        cmap=cmap_freq_offset,
        vmin=slopes_range[0],
        vmax=slopes_range[1],
    )
    plt.imsave(
        RESULTS_FOLDER / f"difference_{echo_direction_name}.png",
        diff,
        dpi=300,
        cmap=cmap_difference,
        vmin=diff_range[0],
        vmax=diff_range[1],
    )

# plot difference between odd and even echoes
diff_odd_even_measured = measured_slopes[0] - measured_slopes[1]
diff_odd_even_measured[~total_mask] = np.nan
diff_odd_even_measured = diff_odd_even_measured[40:-40].T[:, ::-1]
difference_odd_even_image = axes[2, 0].imshow(
    diff_odd_even_measured, cmap=cmap_difference, vmin=diff_range[0], vmax=diff_range[1]
)
axes[2, 0].set_title("Difference Odd-Even")
axes[2, 0].set_axis_off()
plt.imsave(
    RESULTS_FOLDER / "difference_odd_even_measured.png",
    diff_odd_even_measured,
    dpi=300,
    cmap=cmap_difference,
    vmin=diff_range[0],
    vmax=diff_range[1],
)

diff_odd_even_estimated = estimated_slopes[..., 0, 0] - estimated_slopes[..., 0, 1]
diff_odd_even_estimated[~total_mask] = np.nan
diff_odd_even_estimated = diff_odd_even_estimated[40:-40].T[:, ::-1]
difference_odd_even_estimated_image = axes[2, 1].imshow(
    diff_odd_even_estimated,
    cmap=cmap_difference,
    vmin=diff_range[0],
    vmax=diff_range[1],
)
axes[2, 1].set_title("Difference Odd-Even Estimated")
axes[2, 1].set_axis_off()
plt.imsave(
    RESULTS_FOLDER / "difference_odd_even_estimated.png",
    diff_odd_even_estimated,
    dpi=300,
    cmap=cmap_difference,
    vmin=diff_range[0],
    vmax=diff_range[1],
)


f.colorbar(
    estimated_image,
    ax=axes[:, :2],
    orientation="horizontal",
    location="bottom",
    label="Frequency, Hz",
    fraction=0.046,
    pad=0.04,
)
f.colorbar(
    difference_image,
    ax=axes[:, 2],
    orientation="horizontal",
    location="bottom",
    label="Frequency, Hz",
    fraction=0.046,
    pad=0.04,
)

plt.savefig(RESULTS_FOLDER / "summary.png", dpi=300)
