"""Chapter 23.  Within-subject split-half consistency.

Question
--------
For each subject and each block, randomly split that subject's 150 trials into
two halves and compute a classification image (CI) on each half.  The Pearson
correlation between the two CIs measures internal consistency: if the subject
is responding to a stable stimulus feature (whatever it is), the two CIs
should look alike.

Two feature spaces:
    (a) raw centred waveform  (L=2960 dim)
    (b) magnitude spectrum    (F=1481 dim)

We average over n_splits = 200 random splits.  As a baseline we also compute
the same quantity after permuting each subject's response labels (gives a per-
subject null mean).

Outputs
-------
outputs/book3/ch23-split-half/
    ch23_per_subject_splithalf.csv
    ch23_splithalf_hist.png
    ch23_consistency_vs_alignment.png
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
from book3_helpers import (load_all_trials, load_block_waveforms,
                           build_subject_stim_matrix, magnitude_spectrum)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch23-split-half"
OUT_PATH.mkdir(parents = True, exist_ok = True)

N_SPLITS = 200
N_PERM   = 1
RNG      = np.random.default_rng(0)


def split_half_consistency(F, y, n_splits, rng):
    """F: (n_stim, D), y: (n_stim,) in {0,1}.  Average Pearson r between
    half-1-CI and half-2-CI over n_splits random partitions."""
    n = len(y)
    rs = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        h1 = order[:n // 2]; h2 = order[n // 2:]
        for hh in (h1, h2):
            if y[hh].min() == y[hh].max():
                break
        else:
            ci1 = F[h1][y[h1] == 1].mean(0) - F[h1][y[h1] == 0].mean(0)
            ci2 = F[h2][y[h2] == 1].mean(0) - F[h2][y[h2] == 0].mean(0)
            ci1 = ci1 - ci1.mean(); ci2 = ci2 - ci2.mean()
            denom = (np.linalg.norm(ci1) * np.linalg.norm(ci2))
            if denom > 1e-12:
                rs.append(float(np.dot(ci1, ci2) / denom))
    return float(np.mean(rs)) if rs else np.nan


# ============================================================================
# Main
# ============================================================================

trials = load_all_trials()

rows = []
for block in ("full_sentence", "imagined_sentence"):
    print(f"\n[{block}]", flush = True)
    W, stim_nums, lbls, sr = load_block_waveforms(block)
    Y, subjects, sub_stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    idx = {int(n): k for k, n in enumerate(stim_nums)}
    order = np.array([idx[int(n)] for n in sub_stim_nums])
    W_ord = W[order]
    Spec  = magnitude_spectrum(W_ord)

    for s_idx, subj in enumerate(subjects):
        y = Y[s_idx]
        valid = ~np.isnan(y)
        yv = y[valid].astype(int)
        Wv = W_ord[valid]; Sv = Spec[valid]

        r_wave = split_half_consistency(Wv, yv, N_SPLITS, RNG)
        r_spec = split_half_consistency(Sv, yv, N_SPLITS, RNG)

        # null: shuffle responses then compute split-half
        yn = RNG.permutation(yv)
        r_wave_null = split_half_consistency(Wv, yn, N_SPLITS // 2, RNG)
        r_spec_null = split_half_consistency(Sv, yn, N_SPLITS // 2, RNG)

        rows.append(dict(subject = subj, block_type = block,
                         n_yes = int((yv == 1).sum()),
                         n_no  = int((yv == 0).sum()),
                         splithalf_wave = r_wave,
                         splithalf_spec = r_spec,
                         null_wave      = r_wave_null,
                         null_spec      = r_spec_null))
    print(f"  done {len(subjects)} subjects", flush = True)

per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch23_per_subject_splithalf.csv", index = False)

print("\nGroup-level summary of split-half consistency:", flush = True)
print(per_subj.groupby("block_type")[["splithalf_wave", "splithalf_spec",
                                       "null_wave", "null_spec"]].mean(),
      flush = True)

# bring in ch22 alignment for the joint scatter
ci_path = CUR_PATH / "outputs" / "book3" / "ch22-classification-image" / "ch22_per_subject_ci.csv"
if ci_path.exists():
    ci_df = pd.read_csv(ci_path)
    ci_df["subject"]  = ci_df["subject"].astype(str)
    per_subj["subject"] = per_subj["subject"].astype(str)
    merged = per_subj.merge(ci_df[["subject", "block_type", "cos_ci_true"]],
                            on = ["subject", "block_type"])
else:
    merged = None


# ============================================================================
# Plots
# ============================================================================

# Histogram, observed vs null, per block, both feature spaces
fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = "col")
for j, blk in enumerate(("full_sentence", "imagined_sentence")):
    sub = per_subj[per_subj["block_type"] == blk]
    for i, (col, null_col, lbl) in enumerate(
            [("splithalf_wave", "null_wave", "waveform"),
             ("splithalf_spec", "null_spec", "spectrum")]):
        ax = axes[i, j]
        ax.hist(sub[col], bins = 16, color = "#1f77b4", alpha = 0.7,
                edgecolor = "k", label = "observed")
        ax.hist(sub[null_col], bins = 16, color = "#aaaaaa", alpha = 0.6,
                edgecolor = "k", label = "shuffled")
        ax.axvline(sub[col].mean(),       color = "blue",   lw = 2)
        ax.axvline(sub[null_col].mean(),  color = "black",  lw = 2, ls = "--")
        ax.axvline(0, color = "k", lw = 0.6)
        ax.set_xlabel(f"split-half r ({lbl})")
        ax.set_title(f"{blk} -- {lbl}")
        ax.legend(fontsize = 8)
plt.suptitle("Within-subject split-half consistency, observed vs response-shuffled",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch23_splithalf_hist.png", dpi = 130)
plt.close()

# Joint: split-half consistency vs cos(CI_s, CI_true)
if merged is not None:
    fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True, sharex = True)
    colors = {"full_sentence": "#1f77b4", "imagined_sentence": "#d62728"}
    for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
        sub = merged[merged["block_type"] == blk]
        ax.scatter(sub["cos_ci_true"], sub["splithalf_wave"],
                   color = colors[blk], edgecolor = "k", s = 65, alpha = 0.85)
        ax.axhline(0, color = "gray", lw = 0.6)
        ax.axvline(0, color = "gray", lw = 0.6)
        ax.set_xlabel("cos(CI_s, CI_true)  [alignment, ch22]")
        ax.set_ylabel("split-half r (waveform)  [consistency, ch23]")
        ax.set_title(blk)
    plt.suptitle("Internal consistency vs label alignment, per subject",
                 fontsize = 11)
    plt.tight_layout()
    plt.savefig(OUT_PATH / "ch23_consistency_vs_alignment.png", dpi = 130)
    plt.close()

print(f"\nChapter 23 done.  Outputs saved to: {OUT_PATH}", flush = True)
