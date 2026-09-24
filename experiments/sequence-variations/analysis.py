from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa

plt.style.use(["science", "ieee"])


BASE_RESULTS_FOLDER = Path("../../data/results/sequence-variations")
GT_RATIO = 0.25
CI = (-0.056, 0.073)

TRs = [100, 150, 200, 300, 500, 1000]
FAs = [5, 10, 15, 20, 25, 30]
nh_water = 2
nh_acetone = 6
masks = np.load(BASE_RESULTS_FOLDER / "fa10.h5/masks.npy").transpose(2, 0, 1)
masks = masks[:, 40:-40]

errors_tr = []
errors_fa = []
for values, prefix, errors in [(TRs, "tr", errors_tr), (FAs, "fa", errors_fa)]:
    for value in values:
        results_folder = BASE_RESULTS_FOLDER / f"{prefix}{value}.h5"

        c = np.load(results_folder / "c_img_fi.npy")
        c = c[40:-40]
        c[..., 0] /= nh_water
        c[..., 1] /= nh_acetone
        c = np.abs(c) / np.sum(np.abs(c), axis=2, keepdims=True)
        acetone_ratio = c[..., 1]

        for i in range(4):
            estimates = acetone_ratio[masks[i]]
            mean_estimate = np.mean(estimates)
            spatial_variation = np.std(estimates, ddof=1)
            errors.append((GT_RATIO - mean_estimate).item())

cm = 1 / 2.54  # centimeters in inches
ax_width, ax_height = 6 * cm, 4 * cm  # 3.0/383*303, 2.0/383*303

left = 1.0
bottom = 1.0
right = ax_width + left
top = ax_height + bottom

fig_width = right + 0.5
fig_height = top + 0.5

f = plt.figure(figsize=(fig_width, fig_height))
f.subplots_adjust(
    left=left / fig_width,
    right=right / fig_width,
    bottom=bottom / fig_height,
    top=top / fig_height,
)
plt.scatter(
    [tr for tr in TRs for _ in range(4)],
    errors_tr,
    color="#F19C6B",
    marker="o",
    edgecolor="black",
    linewidths=0.5,
    s=5,
)
plt.ylim([-0.2, 0.2])
plt.axhspan(CI[0], CI[1], color="#0A2B5C", alpha=0.2, edgecolor="none", linewidth=0)
plt.axhline(CI[0], linestyle="--", color="#0A2B5C")
plt.text(TRs[-1], CI[0] - 0.01, f"{CI[0]:.3f}", color="black", ha="right", va="top")
plt.text(TRs[-1], CI[1] + 0.01, f"{CI[1]:.3f}", color="black", ha="right", va="bottom")
plt.axhline(CI[1], linestyle="--", color="#0A2B5C")
plt.xlabel("TR, ms")
plt.ylabel("Bias, mol/mol")
plt.savefig(
    BASE_RESULTS_FOLDER / "tr_errors.pdf", bbox_inches="tight", dpi=300, pad_inches=0
)
# plt.show()

f = plt.figure(figsize=(fig_width, fig_height))
f.subplots_adjust(
    left=left / fig_width,
    right=right / fig_width,
    bottom=bottom / fig_height,
    top=top / fig_height,
)
plt.scatter(
    [fa for fa in FAs for _ in range(4)],
    errors_fa,
    color="#F19C6B",
    marker="o",
    edgecolor="black",
    linewidths=0.5,
    s=5,
)
plt.ylim([-0.2, 0.2])
plt.axhspan(CI[0], CI[1], color="#0A2B5C", alpha=0.2, edgecolor="none", linewidth=0)
plt.axhline(CI[0], linestyle="--", color="#0A2B5C")
plt.text(FAs[-1], CI[0] - 0.01, f"{CI[0]:.3f}", color="black", ha="right", va="top")
plt.text(FAs[-1], CI[1] + 0.01, f"{CI[1]:.3f}", color="black", ha="right", va="bottom")
plt.axhline(CI[1], linestyle="--", color="#0A2B5C")
plt.xlabel("Flip angle, °")
plt.ylabel("Bias, mol/mol")
plt.savefig(
    BASE_RESULTS_FOLDER / "fa_errors.pdf", bbox_inches="tight", dpi=300, pad_inches=0
)
# plt.show()
