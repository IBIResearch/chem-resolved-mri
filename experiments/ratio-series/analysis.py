import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from scipy import stats

plt.style.use(["science", "ieee"])

RESULTS_FOLDER = Path("../../data/results/ratio-series")


nominal_acetone_molar_ratios = np.array([0.04, 0.125, 0.25, 0.5, 0.6, 0.8, 1.0])

concentrations = {}

methods = {
    "Proposed": "wo_fi",
    "EPSI": "epsi",
}

concentration_fields = [
    "acetone_c_values_in_mixture",
    "water_c_values_in_mixture",
    "acetone_c_values_in_water",
    "water_c_values_in_water",
]

for method in methods:
    concentrations[method] = {field: [] for field in concentration_fields}

acetone_frequencies = []
water_frequencies = []
for r in nominal_acetone_molar_ratios:
    for method, suffix in methods.items():
        with open(
            RESULTS_FOLDER / f"acetone-{r}/abs_concentrations_of_species_{suffix}.json"
        ) as f:
            d = json.load(f)
            for field in concentration_fields:
                concentrations[method][field].append(np.array(d[field]))

    with open(RESULTS_FOLDER / f"acetone-{r}/model_frequencies.json") as f:
        d = json.load(f)
        acetone_frequencies.append(d["acetone"] / 2 / np.pi * 1000)
        water_frequencies.append(d["water"] / 2 / np.pi * 1000)

df_frequencies = pd.DataFrame(
    {
        "GT ratio": nominal_acetone_molar_ratios,
        "Water Frequency (Hz)": water_frequencies,
        "Acetone Frequency (Hz)": acetone_frequencies,
    }
)

latex_table = df_frequencies.to_latex(index=False, float_format=lambda x: f"{x:.2f}")
print(latex_table)

plt.figure(figsize=(3.5, 2.5))
plt.plot(nominal_acetone_molar_ratios, acetone_frequencies, "o-")
plt.xlabel("Ratio")
plt.ylabel("Acetone Frequency (Hz)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_FOLDER}/acetone_frequencies.png", bbox_inches="tight", dpi=300)
plt.close()

nh_water = 2
nh_acetone = 6
for method in methods:
    ratios_inner_tube = []
    ratios_outer_tube = []
    for i in range(len(concentrations[method]["acetone_c_values_in_mixture"])):
        ratios_inner_tube.append(
            concentrations[method]["acetone_c_values_in_mixture"][i]
            / nh_acetone
            / (
                concentrations[method]["acetone_c_values_in_mixture"][i] / nh_acetone
                + concentrations[method]["water_c_values_in_mixture"][i] / nh_water
            )
        )
        ratios_outer_tube.append(
            concentrations[method]["acetone_c_values_in_water"][i]
            / nh_acetone
            / (
                concentrations[method]["acetone_c_values_in_water"][i] / nh_acetone
                + concentrations[method]["water_c_values_in_water"][i] / nh_water
            )
        )
    concentrations[method]["ratios_inner_tube"] = ratios_inner_tube
    concentrations[method]["ratios_outer_tube"] = ratios_outer_tube


df_inner_parts = []
for i, r in enumerate(nominal_acetone_molar_ratios):
    for method in methods:
        estimates = concentrations[method]["ratios_inner_tube"][i]
        df_inner_parts.append(
            pd.DataFrame(
                {
                    "Ground Truth, mol/mol": [r] * len(estimates),
                    "Estimation, mol/mol": estimates,
                    "Method": [method] * len(estimates),
                }
            )
        )
df_inner = pd.concat(df_inner_parts, ignore_index=True)

f, ax = plt.subplots(figsize=(3.5, 2.5))
sns.boxplot(
    data=df_inner,
    x="Ground Truth, mol/mol",
    y="Estimation, mol/mol",
    hue="Method",
    fill=False,
    ax=ax,
    native_scale=True,
    width=0.75,
    gap=0.25,
    linewidth=0.5,
    fliersize=0.5,
    palette=[
        sns.dark_palette("seagreen")[0],
        sns.dark_palette("seagreen")[-1],
    ],
)

# Add a shared-x anchor: one dashed vertical connector per GT ratio,
# ending at the midpoint of method medians.
median_by_group = (
    df_inner.groupby(["Ground Truth, mol/mol", "Method"])["Estimation, mol/mol"]
    .median()
    .unstack("Method")
)
y_min = ax.get_ylim()[0]
method_order = list(methods.keys())
for i, r in enumerate(nominal_acetone_molar_ratios):
    if r in median_by_group.index and all(
        m in median_by_group.columns for m in method_order
    ):
        midpoint = median_by_group.loc[r, method_order].mean()
        ax.plot(
            [r, r],
            [y_min, midpoint],
            linestyle="--",
            linewidth=0.5,
            color="#999999",
            zorder=0,
            label="shared GT-x anchor" if i == 0 else "_nolegend_",
        )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="expected dependence",
    linewidth=0.5,
    color="#FF4F4F",
)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Ground truth, mol/mol")
plt.savefig(
    f"{RESULTS_FOLDER}/calibration.pdf", bbox_inches="tight", dpi=300, pad_inches=0
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


# Accuracy and precision from signed errors e_i = gt_i - est_i (voxel-wise).
for method in methods:
    df_method = df_inner.loc[df_inner["Method"] == method].copy()

    voxel_errors = (
        df_method["Ground Truth, mol/mol"] - df_method["Estimation, mol/mol"]
    ).to_numpy()

    voxel_stats = signed_error_stats_with_ci(voxel_errors)

    print(f"\n{method} (voxel-level):")
    print(f"  n={voxel_stats['n']}")
    print(
        "  Bias (mean signed error): "
        f"{voxel_stats['mean_err']:.4f} "
        f"[95% CI: {voxel_stats['mean_ci'][0]:.4f}, {voxel_stats['mean_ci'][1]:.4f}]"
    )
    print(
        "  Precision (SD of signed error): "
        f"{voxel_stats['std_err']:.4f} "
        f"[95% CI: {voxel_stats['std_ci'][0]:.4f}, {voxel_stats['std_ci'][1]:.4f}]"
    )


# sensitivity to parametrization
data = []
for i, r in enumerate(nominal_acetone_molar_ratios):
    for offset in range(0, -41, -5):
        with open(
            RESULTS_FOLDER
            / f"acetone-{r}/parametrization-sensitivity/offset{offset}/abs_concentrations_of_species_wo_fi.json"
        ) as f:
            d = json.load(f)
            water = np.array(d["water_c_values_in_mixture"]) / nh_water
            acetone = np.array(d["acetone_c_values_in_mixture"]) / nh_acetone
            estimated_ratio = acetone / (acetone + water)
            gt_ratio = nominal_acetone_molar_ratios[i]
            for j, est_r in enumerate(estimated_ratio):
                data.append([offset, gt_ratio - est_r])
df = pd.DataFrame(data, columns=["Frequency offset, Hz", "Error, mol/mol"])

f, ax = plt.subplots(figsize=(3.5, 2.5))
min_dist = np.min(nominal_acetone_molar_ratios[1:] - nominal_acetone_molar_ratios[:-1])
sns.boxplot(
    data=df,
    x="Frequency offset, Hz",
    y="Error, mol/mol",
    fill=False,
    ax=ax,
    native_scale=True,
    width=0.75 * (min_dist / 1) / (5 / 40) / 2 / 21 * 23,
    gap=0.25,
    linewidth=0.5,
    fliersize=0.5,
    color="#000000",
)
ax.invert_xaxis()
ax.invert_yaxis()
plt.savefig(
    f"{RESULTS_FOLDER}/sensitivity.pdf", bbox_inches="tight", dpi=300, pad_inches=0
)
