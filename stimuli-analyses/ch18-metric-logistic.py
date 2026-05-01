"""Chapter 18.  Trial-level logistic prediction of subject responses.

Question
--------
For each subject and each block, fit a logistic regression that predicts the
binary subject response from a small bank of stimulus-level features.  Compare
the held-out AUC of single-metric models against the experimenter true-label
baseline, and against an "all metrics combined" model.

Features (per stimulus, per block):
    one "block-internal target-minus-distractor contrast" per metric m
    (computed in book2_helpers.block_feature_table).
    A positive contrast means stimulus i is closer (under metric m) to the
    block's other targets than to its distractors.
    +
    trial_index_z  (within-block trial order, z-scored)
    +
    true_target    (the experimenter's label)  -- baseline only

5-fold cross-validation per (subject, block); we report mean held-out ROC AUC.

Outputs
-------
outputs/book2/ch18-logistic/
    ch18_trial_table.csv
    ch18_per_subject_auc.csv
    ch18_metric_summary.csv
    ch18_held_out_auc_box.png
    ch18_per_subject_auc_lines.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import (load_all_trials, load_similarity_matrices,
                           build_stimulus_block_map, block_feature_table,
                           METRICS)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book2" / "ch18-logistic"
OUT_PATH.mkdir(parents = True, exist_ok = True)

N_SPLITS = 5


# ============================================================================
# Build trial-level table with stimulus features
# ============================================================================

print("Loading data ...", flush = True)
trials    = load_all_trials()
sim_dict  = load_similarity_matrices()
block_map = build_stimulus_block_map()

print("Building per-block feature tables ...", flush = True)
feat_per_block = {}
for blk in ("full_sentence", "imagined_sentence"):
    feat_per_block[blk] = block_feature_table(trials, block_map, sim_dict, blk)

# attach features to each trial
parts = []
for blk, feat in feat_per_block.items():
    cols = ["stimulus_number"] + [f"{m}_contrast" for m in METRICS]
    sub  = trials[trials["block_type"] == blk].merge(feat[cols],
                                                     on = "stimulus_number",
                                                     how = "left")
    parts.append(sub)
trial_df = pd.concat(parts, ignore_index = True)

# trial-index normalisation within (subject, block)
trial_df["trial_index_z"] = (trial_df.groupby(["subject", "block_type"])["trial_index"]
                                     .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)))

trial_df.to_csv(OUT_PATH / "ch18_trial_table.csv", index = False)
print(f"  trial_df shape = {trial_df.shape}", flush = True)


# ============================================================================
# Per-subject CV AUC across model families
# ============================================================================

models = {
    "true_label_baseline": ["true_target"],
    "trial_order_only":    ["trial_index_z"],
}
for m in METRICS:
    models[f"single_{m}"] = [f"{m}_contrast", "trial_index_z"]
models["all_metrics"] = [f"{m}_contrast" for m in METRICS] + ["trial_index_z"]

rows = []
print("Per-subject 5-fold CV ...", flush = True)
for (site, subj, blk), sub in trial_df.groupby(["site", "subject", "block_type"]):
    y = sub["called_target"].to_numpy().astype(int)
    if y.min() == y.max():
        continue  # subject was a constant responder in this block
    skf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 0)
    folds = list(skf.split(np.zeros(len(y)), y))
    for name, cols in models.items():
        X = sub[cols].to_numpy().astype(float)
        # standardise non-binary cols
        for j, c in enumerate(cols):
            if c == "true_target":
                continue
            mu, sd = X[:, j].mean(), X[:, j].std() + 1e-9
            X[:, j] = (X[:, j] - mu) / sd
        aucs = []
        for tr, te in folds:
            if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
                continue
            clf = LogisticRegression(max_iter = 1000, C = 1.0)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        if aucs:
            rows.append(dict(site = site, subject = subj, block_type = blk,
                             model = name, n_trials = len(y),
                             cv_auc = float(np.mean(aucs)),
                             cv_auc_std = float(np.std(aucs))))
per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch18_per_subject_auc.csv", index = False)
print(per_subj.head(8).to_string(index = False), flush = True)


# ============================================================================
# Aggregate
# ============================================================================

agg = (per_subj.groupby(["block_type", "model"], as_index = False)
                .agg(n_subjects = ("cv_auc", "size"),
                     mean_auc   = ("cv_auc", "mean"),
                     median_auc = ("cv_auc", "median"),
                     std_auc    = ("cv_auc", "std")))
agg = agg.sort_values(["block_type", "mean_auc"], ascending = [True, False])
agg.to_csv(OUT_PATH / "ch18_metric_summary.csv", index = False)
print("\n=== Per-block, per-model mean held-out AUC ===", flush = True)
print(agg.to_string(index = False), flush = True)


# ============================================================================
# Plots
# ============================================================================

# Box of held-out AUC per model, faceted by block
order = ["true_label_baseline", "trial_order_only"] + \
        [f"single_{m}" for m in METRICS] + ["all_metrics"]
fig, axes = plt.subplots(1, 2, figsize = (14, 6), sharey = True)
for ax, blk in zip(axes, ["full_sentence", "imagined_sentence"]):
    data = [per_subj[(per_subj["block_type"] == blk) & (per_subj["model"] == m)]["cv_auc"].to_numpy()
            for m in order]
    bp = ax.boxplot(data, labels = order, widths = 0.6, patch_artist = True,
                    showfliers = True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1" if blk == "full_sentence" else "#fc9272")
    ax.axhline(0.5, color = "k", lw = 0.6)
    ax.set_title(blk)
    ax.set_ylabel("5-fold held-out AUC")
    ax.tick_params(axis = "x", rotation = 50, labelsize = 8)
plt.suptitle("Held-out AUC of per-subject logistic models, by feature set", fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch18_held_out_auc_box.png", dpi = 130)
plt.close()

# Per-subject paired lines: baseline vs all_metrics
fig, axes = plt.subplots(1, 2, figsize = (12, 6), sharey = True)
for ax, blk in zip(axes, ["full_sentence", "imagined_sentence"]):
    sub = per_subj[per_subj["block_type"] == blk]
    pivot = sub.pivot(index = "subject", columns = "model", values = "cv_auc")
    if "true_label_baseline" not in pivot.columns or "all_metrics" not in pivot.columns:
        continue
    for s in pivot.index:
        ax.plot([0, 1],
                [pivot.loc[s, "true_label_baseline"], pivot.loc[s, "all_metrics"]],
                "o-", color = "gray", alpha = 0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true_label\nbaseline", "all_metrics"])
    ax.axhline(0.5, color = "k", lw = 0.6)
    ax.set_ylabel("CV AUC")
    ax.set_title(blk)
plt.suptitle("Per-subject AUC: experimenter label vs all-metrics combined", fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch18_per_subject_auc_lines.png", dpi = 130)
plt.close()

print(f"\nChapter 18 done.  Outputs saved to: {OUT_PATH}", flush = True)
