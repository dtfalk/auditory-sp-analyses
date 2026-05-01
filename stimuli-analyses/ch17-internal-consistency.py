"""Chapter 17.  Internal response consistency.

Question
--------
Subjects do not see repeated stimuli, so we cannot compute classical test-retest
reliability.  Instead, for each candidate similarity metric M we ask:

    "If two stimuli are M-near-neighbours in this block, did the subject give
     them the same response more often than chance?"

Operationally:
    1. For each block, build a kNN graph on the 150 stimuli using metric M.
    2. For each subject, compute neighbour_consistency =
           mean over (i, j in kNN(i))  of  1[response_i == response_j].
    3. Build a within-subject permutation null (shuffle the subject's responses
       across this block's stimuli, preserving the response rate).

Internal consistency is plotted against label accuracy.  A subject high on
consistency but low on accuracy is plausibly using a stable rule that does not
match the experimenter's labels.

Outputs
-------
outputs/book2/ch17-consistency/
    ch17_per_subject_consistency.csv
    ch17_metric_summary.csv
    ch17_consistency_vs_accuracy.png
    ch17_metric_consistency_box.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import (load_all_trials, load_similarity_matrices,
                           build_subject_stim_matrix, build_stimulus_block_map,
                           METRICS)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book2" / "ch17-consistency"
OUT_PATH.mkdir(parents = True, exist_ok = True)

K_NN    = 10
N_PERM  = 500
RNG     = np.random.default_rng(0)


# ============================================================================
# Helpers
# ============================================================================

def knn_indices(sim_block, k):
    """For each stimulus, return indices of its k nearest neighbours
    (excluding itself), where 'near' = highest similarity.
    sim_block has shape (n_stim, n_stim).
    """
    n = sim_block.shape[0]
    out = np.empty((n, k), dtype = np.int64)
    for i in range(n):
        order = np.argsort(-sim_block[i])  # descending
        nbrs  = order[order != i][:k]
        out[i] = nbrs
    return out


def neighbour_consistency(y, knn):
    """Mean over (i, j in knn[i]) of 1[y[i] == y[j]], ignoring NaN entries."""
    mask = ~np.isnan(y)
    n, k = knn.shape
    same = []
    for i in range(n):
        if not mask[i]:
            continue
        for j in knn[i]:
            if mask[j]:
                same.append(int(y[i] == y[j]))
    return float(np.mean(same)) if same else np.nan


def perm_null_consistency(y, knn, n_perm, rng):
    """Shuffle y (preserving response rate) n_perm times; return null array."""
    null = np.empty(n_perm)
    y_obs = y.copy()
    valid = ~np.isnan(y_obs)
    yv    = y_obs[valid]
    for b in range(n_perm):
        ys = y_obs.copy()
        ys[valid] = rng.permutation(yv)
        null[b] = neighbour_consistency(ys, knn)
    return null


# ============================================================================
# Main
# ============================================================================

print("Loading data ...", flush = True)
trials    = load_all_trials()
sim_dict  = load_similarity_matrices()
block_map = build_stimulus_block_map()

rows = []
metric_summary = []

for block in ("full_sentence", "imagined_sentence"):
    print(f"\n[{block}]", flush = True)
    Y, subjects, stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    print(f"  Y shape = {Y.shape},  n_subjects = {len(subjects)}", flush = True)

    # restrict each metric's matrix to this block's stim set
    for metric in METRICS:
        sim, sim_stim_nums, sim_labels = sim_dict[metric]
        idx = {int(n): k for k, n in enumerate(sim_stim_nums)}
        block_idx = np.array([idx[int(n)] for n in stim_nums])
        sim_block = sim[np.ix_(block_idx, block_idx)]
        knn = knn_indices(sim_block, K_NN)

        # also build a label-based "neighbour" structure as a baseline
        # (for each stim, neighbours = same-label stims, randomly thinned to k)
        same_label = (true_labels[:, None] == true_labels[None, :])
        np.fill_diagonal(same_label, False)
        label_knn = np.empty((len(stim_nums), K_NN), dtype = np.int64)
        for i in range(len(stim_nums)):
            cands = np.where(same_label[i])[0]
            label_knn[i] = RNG.choice(cands, size = K_NN, replace = False)

        cons_vals  = []
        for s_idx, subj in enumerate(subjects):
            y = Y[s_idx]
            obs = neighbour_consistency(y, knn)
            null = perm_null_consistency(y, knn, N_PERM, RNG)
            p_perm = (1 + np.sum(null >= obs)) / (N_PERM + 1)
            obs_lab = neighbour_consistency(y, label_knn)
            cons_vals.append(obs)
            rows.append(dict(subject = subj, block_type = block, metric = metric,
                             k = K_NN,
                             neighbour_consistency = obs,
                             null_mean = float(np.mean(null)),
                             null_std  = float(np.std(null)),
                             p_perm    = p_perm,
                             label_neighbour_consistency = obs_lab,
                             accuracy = float(np.nanmean(y == true_labels))))
        metric_summary.append(dict(block_type = block, metric = metric,
                                   mean_consistency = float(np.mean(cons_vals)),
                                   median_consistency = float(np.median(cons_vals)),
                                   n_subjects = len(subjects)))
        print(f"  {metric:18s}  mean cons = {np.mean(cons_vals):.3f}",
              flush = True)

per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch17_per_subject_consistency.csv", index = False)
ms = pd.DataFrame(metric_summary)
ms.to_csv(OUT_PATH / "ch17_metric_summary.csv", index = False)


# ============================================================================
# Plots
# ============================================================================

# Consistency vs accuracy scatter, faceted by metric, coloured by block
fig, axes = plt.subplots(1, len(METRICS), figsize = (4 * len(METRICS), 4.2),
                         sharey = True, sharex = True)
colors = {"full_sentence": "#1f77b4", "imagined_sentence": "#d62728"}
for ax, metric in zip(axes, METRICS):
    sub = per_subj[per_subj["metric"] == metric]
    for blk, g in sub.groupby("block_type"):
        ax.scatter(g["accuracy"], g["neighbour_consistency"],
                   color = colors[blk], edgecolor = "k", s = 45,
                   alpha = 0.85, label = blk)
    ax.axhline(0.5, color = "gray", lw = 0.7)
    ax.axvline(0.5, color = "gray", lw = 0.7)
    ax.set_title(metric, fontsize = 9)
    ax.set_xlabel("Label accuracy")
axes[0].set_ylabel(f"k={K_NN} neighbour consistency")
axes[0].legend(fontsize = 7, loc = "lower right")
plt.suptitle("Internal consistency vs label accuracy, per (subject, metric, block)",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch17_consistency_vs_accuracy.png", dpi = 130)
plt.close()

# Per-metric box of consistency
fig, ax = plt.subplots(figsize = (10, 5))
positions = []
data = []
labels_x = []
i = 0
for metric in METRICS:
    for blk in ("full_sentence", "imagined_sentence"):
        sub = per_subj[(per_subj["metric"] == metric) & (per_subj["block_type"] == blk)]
        data.append(sub["neighbour_consistency"].to_numpy())
        positions.append(i)
        labels_x.append(f"{metric}\n{blk}")
        i += 1
    i += 0.7
bp = ax.boxplot(data, positions = positions, widths = 0.7,
                showfliers = True, patch_artist = True)
for patch, lbl in zip(bp["boxes"], labels_x):
    patch.set_facecolor("#9ecae1" if "full" in lbl else "#fc9272")
ax.axhline(0.5, color = "k", lw = 0.6)
ax.set_xticks(positions)
ax.set_xticklabels(labels_x, rotation = 65, fontsize = 7, ha = "right")
ax.set_ylabel(f"k={K_NN} neighbour consistency")
ax.set_title("Distribution of internal consistency across subjects, per metric x block")
plt.tight_layout()
plt.savefig(OUT_PATH / "ch17_metric_consistency_box.png", dpi = 130)
plt.close()

print(f"\nChapter 17 done.  Outputs saved to: {OUT_PATH}", flush = True)
