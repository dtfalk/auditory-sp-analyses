"""Chapter 30.  Reminder-cycle x block-position temporal tests.

Builds on chapter 25 but adds the within-subject categorical and
interaction tests that ch25 left as descriptive plots.

Per block, per subject we compute:

    bin3   = trials_since_reminder bucket  (first5 / mid5 / last5)
    third  = block-third by trial_index    (early / mid / late)

Tests
-----
(A) Within-subject Friedman test on (response_rate, hit_rate, fa_rate,
    accuracy) across the 3 bin3 levels.  Same again across the 3
    third levels.
(B) 3 x 3 cell means (bin3 x third) per subject -> within-subject
    Wilcoxon on a couple of orthogonal contrasts to look for an
    interaction:
        - "fatigue minus refresh"  = (third=late, bin3=last5)
                                    - (third=early, bin3=first5)
        - "early-cycle drift"      = (third=late, bin3=first5)
                                    - (third=early, bin3=first5)
        - "late-cycle drift"       = (third=late, bin3=last5)
                                    - (third=early, bin3=last5)
    Each contrast is computed per subject per block then Wilcoxon vs 0.
(C) Per-subject logistic with the explicit interaction:
        called_target ~ z_tsr + z_pos + z_tsr:z_pos
                      + true * (z_tsr + z_pos + z_tsr:z_pos)
                      + true
    Group Wilcoxon vs 0 on every coefficient; in particular on the
    new beta_tsr_x_pos and beta_tsr_x_pos_x_truth.

Outputs
-------
outputs/book3/ch30-temporal-interactions/
    ch30_friedman_bin3.csv
    ch30_friedman_third.csv
    ch30_per_subject_cell_means.csv
    ch30_contrast_wilcoxon.csv
    ch30_per_subject_slopes.csv
    ch30_group_slope_tests.csv
    ch30_cell_heatmap.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats           import friedmanchisquare, wilcoxon
from sklearn.linear_model  import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import load_all_trials

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch30-temporal-interactions"
OUT_PATH.mkdir(parents = True, exist_ok = True)


# ============================================================================
# Load and augment
# ============================================================================

trials = load_all_trials()
trials = trials.sort_values(["site", "subject", "block_type",
                              "trial_index"]).copy()
trials["tsr"] = trials["trial_index"] % 15
trials["pos"] = trials["trial_index"].astype(float)
trials["bin3"]  = pd.cut(trials["tsr"],
                          bins   = [-0.1, 4.5, 9.5, 14.1],
                          labels = ["first5", "mid5", "last5"])
trials["third"] = pd.cut(trials["pos"],
                          bins   = [-0.1, 49.5, 99.5, 149.1],
                          labels = ["early", "mid", "late"])
trials["called_target"] = trials["called_target"].astype(float)
trials["true_target"]   = trials["true_target"].astype(float)
trials["hit"]  = ((trials["true_target"] == 1)
                  & (trials["called_target"] == 1)).astype(float)
trials["miss"] = ((trials["true_target"] == 1)
                  & (trials["called_target"] == 0)).astype(float)
trials["fa"]   = ((trials["true_target"] == 0)
                  & (trials["called_target"] == 1)).astype(float)
trials["correct"] = (trials["called_target"]
                     == trials["true_target"]).astype(float)


# ============================================================================
# (A) Friedman tests on bin3 and on third
# ============================================================================

def per_subject_means(grouper):
    """Return wide df: rows = subject x block_type, cols = grouper levels,
    one row per (response_rate, hit_rate, fa_rate, accuracy)."""
    out = []
    for (subj, blk), g in trials.groupby(["subject", "block_type"]):
        for level, gl in g.groupby(grouper, observed = True):
            out.append(dict(subject = subj, block_type = blk,
                             level   = str(level),
                             response_rate = gl["called_target"].mean(),
                             hit_rate      = (gl[gl["true_target"] == 1]
                                              ["called_target"].mean()),
                             fa_rate       = (gl[gl["true_target"] == 0]
                                              ["called_target"].mean()),
                             accuracy      = gl["correct"].mean()))
    return pd.DataFrame(out)


def friedman_table(grouper, level_order):
    long_df = per_subject_means(grouper)
    rows = []
    for blk in ("full_sentence", "imagined_sentence"):
        sub = long_df[long_df["block_type"] == blk]
        for metric in ("response_rate", "hit_rate", "fa_rate", "accuracy"):
            wide = (sub.pivot(index = "subject", columns = "level",
                               values = metric)
                       .reindex(columns = level_order)
                       .dropna())
            stat, p = friedmanchisquare(*[wide[c].values
                                           for c in level_order])
            row = dict(block_type = blk, metric = metric,
                        n_subj = len(wide),
                        chi2 = float(stat), p = float(p))
            for c in level_order:
                row[f"mean_{c}"] = float(wide[c].mean())
            rows.append(row)
    return pd.DataFrame(rows)


bin3_df = friedman_table("bin3", ["first5", "mid5", "last5"])
bin3_df.to_csv(OUT_PATH / "ch30_friedman_bin3.csv", index = False)
print("=== Friedman on bin3 (first5/mid5/last5 within reminder cycle) ===",
      flush = True)
print(bin3_df.round(4).to_string(index = False), flush = True)

third_df = friedman_table("third", ["early", "mid", "late"])
third_df.to_csv(OUT_PATH / "ch30_friedman_third.csv", index = False)
print("\n=== Friedman on block-third (early/mid/late within block) ===",
      flush = True)
print(third_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# (B) bin3 x third cell means and interaction contrasts
# ============================================================================

cell_rows = []
for (subj, blk), g in trials.groupby(["subject", "block_type"]):
    for b in ("first5", "mid5", "last5"):
        for t in ("early", "mid", "late"):
            cell = g[(g["bin3"] == b) & (g["third"] == t)]
            if len(cell) == 0:
                continue
            cell_rows.append(dict(subject = subj, block_type = blk,
                                   bin3 = b, third = t, n = len(cell),
                                   response_rate = cell["called_target"].mean(),
                                   accuracy      = cell["correct"].mean(),
                                   hit_rate      = (cell[cell["true_target"] == 1]
                                                    ["called_target"].mean()),
                                   fa_rate       = (cell[cell["true_target"] == 0]
                                                    ["called_target"].mean())))
cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(OUT_PATH / "ch30_per_subject_cell_means.csv", index = False)


def cell_value(d, b, t, metric):
    sub = d[(d["bin3"] == b) & (d["third"] == t)]
    return float(sub[metric].iloc[0]) if len(sub) == 1 else np.nan


contrast_rows = []
for (subj, blk), g in cell_df.groupby(["subject", "block_type"]):
    for metric in ("response_rate", "accuracy", "hit_rate", "fa_rate"):
        contrast_rows.append(dict(
            subject = subj, block_type = blk, metric = metric,
            fatigue_minus_refresh = cell_value(g, "last5",  "late",  metric)
                                     - cell_value(g, "first5", "early", metric),
            early_cycle_drift     = cell_value(g, "first5", "late",  metric)
                                     - cell_value(g, "first5", "early", metric),
            late_cycle_drift      = cell_value(g, "last5",  "late",  metric)
                                     - cell_value(g, "last5",  "early", metric),
            late_minus_first_in_late_third
                                  = cell_value(g, "last5",  "late",  metric)
                                     - cell_value(g, "first5", "late",  metric)))
contrast_df = pd.DataFrame(contrast_rows)

wilc_rows = []
for blk in ("full_sentence", "imagined_sentence"):
    sub = contrast_df[contrast_df["block_type"] == blk]
    for metric in ("response_rate", "accuracy", "hit_rate", "fa_rate"):
        s = sub[sub["metric"] == metric]
        for col in ("fatigue_minus_refresh", "early_cycle_drift",
                     "late_cycle_drift", "late_minus_first_in_late_third"):
            v = s[col].dropna().values
            if len(v) < 6 or np.allclose(v, 0):
                W, p = (np.nan, np.nan)
            else:
                W, p = wilcoxon(v)
            wilc_rows.append(dict(block_type = blk, metric = metric,
                                   contrast = col, n = len(v),
                                   median = float(np.median(v)),
                                   mean   = float(np.mean(v)),
                                   W = (float(W) if not np.isnan(W) else np.nan),
                                   p = (float(p) if not np.isnan(p) else np.nan)))
wilc_df = pd.DataFrame(wilc_rows)
wilc_df.to_csv(OUT_PATH / "ch30_contrast_wilcoxon.csv", index = False)
print("\n=== Within-subject contrast Wilcoxon vs 0 ===", flush = True)
print(wilc_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# (C) Per-subject logistic with tsr x pos interaction
# ============================================================================

slope_rows = []
for (subj, blk), g in trials.groupby(["subject", "block_type"]):
    z_tsr = (g["tsr"] - g["tsr"].mean()) / g["tsr"].std()
    z_pos = (g["pos"] - g["pos"].mean()) / g["pos"].std()
    tt    = g["true_target"].values
    z_int = (z_tsr * z_pos).values
    X = np.column_stack([z_tsr.values, z_pos.values, z_int,
                          tt,
                          tt * z_tsr.values, tt * z_pos.values, tt * z_int])
    y = g["called_target"].values
    if len(np.unique(y)) < 2:
        continue
    lr = LogisticRegression(C = 1.0, solver = "liblinear",
                             max_iter = 2000, fit_intercept = True)
    lr.fit(X, y)
    b = lr.coef_.ravel()
    slope_rows.append(dict(subject = subj, block_type = blk,
                            beta_tsr               = b[0],
                            beta_pos               = b[1],
                            beta_tsr_x_pos         = b[2],
                            beta_truetarget        = b[3],
                            beta_tsr_x_truth       = b[4],
                            beta_pos_x_truth       = b[5],
                            beta_tsr_x_pos_x_truth = b[6],
                            intercept              = float(lr.intercept_[0])))
slope_df = pd.DataFrame(slope_rows)
slope_df.to_csv(OUT_PATH / "ch30_per_subject_slopes.csv", index = False)

beta_cols = ["beta_tsr", "beta_pos", "beta_tsr_x_pos",
              "beta_truetarget", "beta_tsr_x_truth", "beta_pos_x_truth",
              "beta_tsr_x_pos_x_truth"]
group_rows = []
for blk in ("full_sentence", "imagined_sentence"):
    sub = slope_df[slope_df["block_type"] == blk]
    for col in beta_cols:
        v = sub[col].dropna().values
        W, p = wilcoxon(v)
        group_rows.append(dict(block_type = blk, beta = col,
                                median = float(np.median(v)),
                                mean   = float(np.mean(v)),
                                n = len(v),
                                W = float(W), p = float(p)))
group_df = pd.DataFrame(group_rows)
group_df.to_csv(OUT_PATH / "ch30_group_slope_tests.csv", index = False)
print("\n=== Group Wilcoxon on per-subject logistic slopes (vs 0) ===",
      flush = True)
print(group_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# (D) Heatmap of mean response_rate over (bin3 x third), per block
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize = (10, 4))
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = (cell_df[cell_df["block_type"] == blk]
            .groupby(["bin3", "third"], observed = True)["response_rate"]
            .mean()
            .unstack("third")
            .reindex(index   = ["first5", "mid5", "last5"],
                     columns = ["early",  "mid",  "late"]))
    im = ax.imshow(sub.values, cmap = "viridis", aspect = "auto",
                    vmin = 0.4, vmax = 0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(sub.columns)
    ax.set_yticks(range(3)); ax.set_yticklabels(sub.index)
    ax.set_xlabel("block-third");  ax.set_ylabel("trials-since-reminder")
    ax.set_title(blk)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{sub.values[i, j]:.3f}",
                     ha = "center", va = "center", color = "white",
                     fontsize = 9)
    plt.colorbar(im, ax = ax, fraction = 0.045)
plt.suptitle("Mean response rate (bin3 x block-third) -- group mean")
plt.tight_layout()
plt.savefig(OUT_PATH / "ch30_cell_heatmap.png", dpi = 130)
plt.close()

print(f"\nChapter 30 done.  Outputs: {OUT_PATH}", flush = True)
