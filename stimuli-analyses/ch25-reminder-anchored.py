"""Chapter 25.  Reminder-anchored and block-position effects, within subject.

Each of the two blocks has 150 trials with a periodic reminder screen
(re-playing the wall.wav template) inserted before trial 15, 30, 45, ..., 135.
Define for each trial (0-indexed within block):
    trials_since_reminder  = trial_index % 15        in {0,..,14}
    reminder_segment       = trial_index // 15       in {0,..,9}
    pos_in_block           = trial_index             in {0,..,149}
    bin5  = "first5"  if trials_since_reminder < 5
            "mid5"   if 5 <= trials_since_reminder < 10
            "last5"  if trials_since_reminder >= 10

We do three within-subject analyses per block:
    (A) Per-subject hit / FA / response / accuracy as a function of bin5.
    (B) Per-subject logistic regression:
            called_target ~ z(trials_since_reminder)
                          + true_target * z(trials_since_reminder)
                          + z(pos_in_block)
                          + true_target * z(pos_in_block)
        Per-subject slope estimates -> group Wilcoxon vs 0.
    (C) Per-subject within-segment slope of mean response rate over
        reminder_segment (early vs late in block).

Outputs
-------
outputs/book3/ch25-reminder-anchored/
    ch25_per_subject_bin5.csv
    ch25_per_subject_slopes.csv
    ch25_bin5_group.png
    ch25_slope_forest.png
    ch25_segment_drift.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats           import wilcoxon
from sklearn.linear_model  import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book3_helpers import load_all_trials

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
OUT_PATH.mkdir(parents = True, exist_ok = True)

REMINDER_PERIOD = 15


# ============================================================================
# Augment trials with temporal covariates
# ============================================================================

trials = load_all_trials()
trials = trials.sort_values(["site", "subject", "block_type", "trial_index"]).copy()
trials["trials_since_reminder"] = trials["trial_index"] % REMINDER_PERIOD
trials["reminder_segment"]      = trials["trial_index"] // REMINDER_PERIOD
trials["pos_in_block"]          = trials["trial_index"].astype(float)
trials["bin5"] = pd.cut(trials["trials_since_reminder"],
                        bins = [-0.1, 4.5, 9.5, 14.1],
                        labels = ["first5", "mid5", "last5"])
trials["called_target"] = trials["called_target"].astype(float)
trials["true_target"]   = trials["true_target"].astype(float)


# ============================================================================
# (A) Per-subject mean response/hit/FA/accuracy by bin5
# ============================================================================

bin5_rows = []
for (subj, blk), g in trials.groupby(["subject", "block_type"]):
    for bin5, gb in g.groupby("bin5", observed = True):
        if len(gb) == 0:
            continue
        n = len(gb)
        resp = gb["called_target"].mean()
        is_t = gb["true_target"] == 1
        is_d = gb["true_target"] == 0
        hit  = gb.loc[is_t, "called_target"].mean() if is_t.any() else np.nan
        fa   = gb.loc[is_d, "called_target"].mean() if is_d.any() else np.nan
        acc  = (gb["called_target"] == gb["true_target"]).mean()
        bin5_rows.append(dict(subject = subj, block_type = blk, bin5 = str(bin5),
                              n = n, response_rate = resp, hit_rate = hit,
                              fa_rate = fa, accuracy = acc))
bin5_df = pd.DataFrame(bin5_rows)
bin5_df.to_csv(OUT_PATH / "ch25_per_subject_bin5.csv", index = False)


# ============================================================================
# (B) Per-subject logistic with temporal covariates
# ============================================================================

slope_rows = []
for (subj, blk), g in trials.groupby(["subject", "block_type"]):
    g = g.copy()
    if len(g) < 30 or g["called_target"].nunique() < 2:
        continue
    z_tsr = (g["trials_since_reminder"] - g["trials_since_reminder"].mean()) / \
             g["trials_since_reminder"].std(ddof = 0)
    z_pos = (g["pos_in_block"] - g["pos_in_block"].mean()) / \
             g["pos_in_block"].std(ddof = 0)
    tt    = g["true_target"].values
    X = np.column_stack([z_tsr, tt, z_tsr * tt, z_pos, z_pos * tt])
    y = g["called_target"].values.astype(int)
    clf = LogisticRegression(C = 1.0, solver = "lbfgs", max_iter = 1000)
    clf.fit(X, y)
    b = clf.coef_.ravel()
    slope_rows.append(dict(subject = subj, block_type = blk,
                           beta_tsr           = b[0],
                           beta_truetarget    = b[1],
                           beta_tsr_x_truth   = b[2],
                           beta_pos           = b[3],
                           beta_pos_x_truth   = b[4],
                           intercept          = float(clf.intercept_[0])))
slope_df = pd.DataFrame(slope_rows)
slope_df.to_csv(OUT_PATH / "ch25_per_subject_slopes.csv", index = False)

print("\nGroup-level Wilcoxon (per-subject slopes vs 0):", flush = True)
beta_cols = ["beta_tsr", "beta_truetarget", "beta_tsr_x_truth",
             "beta_pos", "beta_pos_x_truth"]
group_summary = []
for blk, g in slope_df.groupby("block_type"):
    for col in beta_cols:
        v = g[col].dropna().values
        if len(v) < 5:
            continue
        stat, p = wilcoxon(v)
        group_summary.append(dict(block_type = blk, beta = col,
                                  median = float(np.median(v)),
                                  mean   = float(v.mean()),
                                  n      = len(v),
                                  W      = float(stat),
                                  p      = float(p)))
group_df = pd.DataFrame(group_summary)
print(group_df.to_string(index = False), flush = True)
group_df.to_csv(OUT_PATH / "ch25_group_slope_tests.csv", index = False)


# ============================================================================
# (C) Per-subject response-rate drift across the 10 reminder segments
# ============================================================================

seg_rows = []
for (subj, blk), g in trials.groupby(["subject", "block_type"]):
    seg_means = g.groupby("reminder_segment")["called_target"].mean()
    if len(seg_means) < 5:
        continue
    x = seg_means.index.values.astype(float)
    y = seg_means.values.astype(float)
    slope = np.polyfit(x, y, 1)[0]
    seg_rows.append(dict(subject = subj, block_type = blk,
                         drift_slope = slope,
                         first_seg_mean = float(y[0]),
                         last_seg_mean  = float(y[-1])))
seg_df = pd.DataFrame(seg_rows)
seg_df.to_csv(OUT_PATH / "ch25_per_subject_segment_drift.csv", index = False)


# ============================================================================
# Plots
# ============================================================================

# (A) bin5 group plot: hit / FA / response rate vs bin5 (mean +/- SE across subjects)
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
order_bins = ["first5", "mid5", "last5"]
metrics = [("response_rate", "tab:blue", "P(yes)"),
           ("hit_rate",      "tab:green", "hit rate"),
           ("fa_rate",       "tab:red",   "false-alarm rate"),
           ("accuracy",      "tab:purple","accuracy")]
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = bin5_df[bin5_df["block_type"] == blk]
    x = np.arange(3)
    for col, color, label in metrics:
        m = sub.groupby("bin5", observed = True)[col].mean().reindex(order_bins)
        s = sub.groupby("bin5", observed = True)[col].sem().reindex(order_bins)
        ax.errorbar(x, m.values, yerr = s.values, marker = "o", lw = 1.5,
                    capsize = 3, label = label, color = color)
    ax.set_xticks(x); ax.set_xticklabels(order_bins)
    ax.set_xlabel("trials since reminder")
    ax.set_title(blk)
    ax.axhline(0.5, color = "gray", lw = 0.6, ls = "--")
    ax.set_ylim(0, 1)
    if blk == "full_sentence":
        ax.set_ylabel("rate")
    ax.legend(fontsize = 8, loc = "lower right")
plt.suptitle("Reminder-anchored time course: first 5 / middle 5 / last 5 trials"
             " after each reminder", fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch25_bin5_group.png", dpi = 130)
plt.close()

# (B) Forest plot of per-subject slopes
fig, axes = plt.subplots(1, 2, figsize = (14, 7), sharex = True)
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = slope_df[slope_df["block_type"] == blk]
    ys = np.arange(len(beta_cols))
    means = [sub[c].mean() for c in beta_cols]
    serr  = [sub[c].sem()  for c in beta_cols]
    # individual points (jittered)
    for i, c in enumerate(beta_cols):
        v = sub[c].dropna().values
        ax.scatter(v, np.full_like(v, i, dtype = float)
                       + np.random.uniform(-0.15, 0.15, size = len(v)),
                   color = "lightgray", s = 25, edgecolor = "k", alpha = 0.7)
    ax.errorbar(means, ys, xerr = serr, fmt = "o", color = "tab:blue",
                ecolor = "tab:blue", capsize = 4, lw = 2, ms = 8,
                label = "mean +/- SE")
    ax.axvline(0, color = "k", lw = 0.7)
    ax.set_yticks(ys); ax.set_yticklabels(beta_cols)
    ax.set_xlabel("logistic slope")
    ax.set_title(blk)
plt.suptitle("Per-subject temporal-covariate slopes (gray = subjects, blue = group)",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch25_slope_forest.png", dpi = 130)
plt.close()

# (C) Segment drift histogram
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = seg_df[seg_df["block_type"] == blk]
    ax.hist(sub["drift_slope"], bins = 16, color = "tab:orange",
            edgecolor = "k", alpha = 0.85)
    ax.axvline(0, color = "k", lw = 0.7)
    ax.axvline(sub["drift_slope"].mean(), color = "blue", lw = 2,
               label = f"mean = {sub['drift_slope'].mean():+.4f}")
    ax.set_xlabel("response-rate slope across 10 reminder segments")
    ax.set_title(blk)
    ax.legend()
plt.suptitle("Per-subject drift in response rate across the block (segment-level)",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch25_segment_drift.png", dpi = 130)
plt.close()

print(f"\nChapter 25 done.  Outputs saved to: {OUT_PATH}", flush = True)
