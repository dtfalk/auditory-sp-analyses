"""Chapter 22.  Per-subject classification image (yes-minus-no waveform template).

Question
--------
Each subject's responses define an internal "yes" template:
    CI_s = mean( waveforms of stimuli they called target )
         - mean( waveforms of stimuli they called distractor )
Compare CI_s to the experimenter-defined ground-truth template
    CI_true = mean( target waveforms ) - mean( distractor waveforms )
in cosine similarity.  Also compare CI_s to the original wall.wav template t.

Per-subject (per block) we report:
    cos(CI_s, CI_true), cos(CI_s, t_wall),
    norm(CI_s), p_perm via response-shuffle null.

Outputs
-------
outputs/book3/ch22-classification-image/
    ch22_per_subject_ci.csv
    ch22_alignment_hist.png
    ch22_ci_examples.png
    ch22_ci_vs_wall_scatter.png
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
from book3_helpers import (load_all_trials, load_block_waveforms, load_template,
                           build_subject_stim_matrix)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch22-classification-image"
OUT_PATH.mkdir(parents = True, exist_ok = True)

N_PERM = 1000
RNG    = np.random.default_rng(0)


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def classification_image(W, y):
    """W: (n_stim, L), y in {0,1,nan}.  Returns mean(yes) - mean(no) waveform."""
    valid = ~np.isnan(y)
    yv = y[valid].astype(int)
    Wv = W[valid]
    if (yv == 1).sum() == 0 or (yv == 0).sum() == 0:
        return None
    return Wv[yv == 1].mean(axis = 0) - Wv[yv == 0].mean(axis = 0)


# ============================================================================
# Main
# ============================================================================

t_wall, sr = load_template()
print(f"wall.wav: L={len(t_wall)}, sr={sr}", flush = True)

trials = load_all_trials()

rows = []
example_subjects = {}     # for the figure: best/worst alignment per block

for block in ("full_sentence", "imagined_sentence"):
    print(f"\n[{block}]", flush = True)
    W, stim_nums, lbls, _ = load_block_waveforms(block)
    L = W.shape[1]
    # truncate template to block length (should already match)
    t_b = t_wall[:L].copy(); t_b -= t_b.mean()
    Y, subjects, sub_stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    # align stimulus order between W and Y
    idx = {int(n): k for k, n in enumerate(stim_nums)}
    order = np.array([idx[int(n)] for n in sub_stim_nums])
    W_ord = W[order]
    lbls_ord = lbls[order]
    assert np.array_equal(lbls_ord, true_labels)

    CI_true = W_ord[lbls_ord == 1].mean(axis = 0) - W_ord[lbls_ord == 0].mean(axis = 0)
    print(f"  CI_true: ||.||={np.linalg.norm(CI_true):.2f}, "
          f"cos(CI_true, wall)={cos(CI_true, t_b):+.4f}", flush = True)

    block_rows = []
    for s_idx, subj in enumerate(subjects):
        y = Y[s_idx]
        CI_s = classification_image(W_ord, y)
        if CI_s is None:
            continue
        c_true = cos(CI_s, CI_true)
        c_wall = cos(CI_s, t_b)

        # permutation null: shuffle this subject's response vector across stims
        valid = ~np.isnan(y)
        yv = y[valid].astype(int)
        Wv = W_ord[valid]
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            yp = RNG.permutation(yv)
            CIp = Wv[yp == 1].mean(axis = 0) - Wv[yp == 0].mean(axis = 0)
            null[b] = cos(CIp, CI_true)
        p_two = (1 + np.sum(np.abs(null) >= np.abs(c_true))) / (N_PERM + 1)
        p_pos = (1 + np.sum(null >= c_true)) / (N_PERM + 1)

        block_rows.append(dict(subject = subj, block_type = block,
                               cos_ci_true = c_true,
                               cos_ci_wall = c_wall,
                               ci_norm = float(np.linalg.norm(CI_s)),
                               null_mean = float(null.mean()),
                               null_std  = float(null.std()),
                               p_perm_two = p_two,
                               p_perm_pos = p_pos,
                               n_yes = int((yv == 1).sum()),
                               n_no  = int((yv == 0).sum())))
    rows.extend(block_rows)
    bdf = pd.DataFrame(block_rows).sort_values("cos_ci_true")
    example_subjects[block] = (bdf.iloc[0]["subject"], bdf.iloc[-1]["subject"],
                               W_ord, Y, subjects, CI_true, t_b)

per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch22_per_subject_ci.csv", index = False)
print("\nGroup means of cos(CI_s, CI_true):", flush = True)
print(per_subj.groupby("block_type")["cos_ci_true"].agg(["mean", "median", "std",
                                                          lambda x: (x > 0).mean()]),
      flush = True)
print("\n#subjects with p_perm_two < 0.05:", flush = True)
print(per_subj.groupby("block_type")["p_perm_two"].apply(lambda s: int((s < 0.05).sum())),
      flush = True)


# ============================================================================
# Plots
# ============================================================================

# Histogram of alignment, with null-mean reference
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
colors = {"full_sentence": "#1f77b4", "imagined_sentence": "#d62728"}
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = per_subj[per_subj["block_type"] == blk]
    ax.hist(sub["cos_ci_true"], bins = 18, color = colors[blk],
            edgecolor = "k", alpha = 0.85)
    ax.axvline(0, color = "k", lw = 0.7)
    ax.axvline(sub["cos_ci_true"].mean(), color = "orange", lw = 2,
               label = f"mean = {sub['cos_ci_true'].mean():+.3f}")
    ax.set_xlabel("cos(CI_s, CI_true)")
    ax.set_title(f"{blk}: per-subject classification-image alignment")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_PATH / "ch22_alignment_hist.png", dpi = 130)
plt.close()

# Example CI waveforms: best/worst aligned subject per block + CI_true overlay
fig, axes = plt.subplots(2, 2, figsize = (14, 7), sharex = True)
for col, blk in enumerate(("full_sentence", "imagined_sentence")):
    worst_subj, best_subj, W_ord, Y, subjects, CI_true, t_b = example_subjects[blk]
    s2i = {s: i for i, s in enumerate(subjects)}
    for row, (label, subj) in enumerate([("worst-aligned", worst_subj),
                                         ("best-aligned",  best_subj)]):
        ax = axes[row, col]
        y  = Y[s2i[subj]]
        CI = classification_image(W_ord, y)
        ax.plot(CI / (np.linalg.norm(CI) + 1e-12), color = "tab:blue",
                lw = 0.6, label = f"CI_s ({label}, subj {subj[-3:]})")
        ax.plot(CI_true / (np.linalg.norm(CI_true) + 1e-12), color = "k",
                lw = 0.6, alpha = 0.7, label = "CI_true")
        ax.set_ylabel("normalised amplitude")
        ax.legend(fontsize = 7, loc = "upper right")
        if row == 0:
            ax.set_title(blk)
axes[1, 0].set_xlabel("sample index")
axes[1, 1].set_xlabel("sample index")
plt.tight_layout()
plt.savefig(OUT_PATH / "ch22_ci_examples.png", dpi = 130)
plt.close()

# Scatter: per-subject cos(CI_s, CI_true)  vs  cos(CI_s, wall)
fig, ax = plt.subplots(figsize = (7, 6))
for blk, g in per_subj.groupby("block_type"):
    ax.scatter(g["cos_ci_wall"], g["cos_ci_true"], color = colors[blk],
               edgecolor = "k", s = 70, alpha = 0.85, label = blk)
ax.axhline(0, color = "gray", lw = 0.6)
ax.axvline(0, color = "gray", lw = 0.6)
ax.set_xlabel("cos(CI_s, wall.wav)")
ax.set_ylabel("cos(CI_s, CI_true)")
ax.set_title("Per-subject CI alignment: vs. label-template (y) and vs. wall (x)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PATH / "ch22_ci_vs_wall_scatter.png", dpi = 130)
plt.close()

print(f"\nChapter 22 done.  Outputs saved to: {OUT_PATH}", flush = True)
