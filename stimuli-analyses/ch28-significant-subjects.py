"""Chapter 28.  Deep dive on individually-significant subjects.

Pull the subjects who individually surpass per-subject permutation p<0.05
in any of: Ch.16 d', Ch.22 cos_ci_true, Ch.24 cos_w_true.  For each such
(subject, block_type), report:
    - all four metrics
    - hit / FA / accuracy / criterion
    - reminder-anchored bin5 means
    - questionnaire scores
    - is the same subject significant in BOTH blocks?

Also build a short table of "stably aligned subjects" defined as those who
are positive on cos_ci_true AND cos_w_true AND beta_truetarget in a given
block.

Outputs
-------
outputs/book3/ch28-significant-subjects/
    ch28_significance_flags.csv
    ch28_significant_table.csv
    ch28_stably_aligned.csv
    ch28_signif_overlap_venn.png
    ch28_significant_metric_table.png
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
from book3_helpers import load_questionnaires

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch28-significant-subjects"
OUT_PATH.mkdir(parents = True, exist_ok = True)

ALPHA = 0.05


# ============================================================================
# Pull
# ============================================================================

dp = pd.read_csv(CUR_PATH / "outputs" / "book2" / "ch16-sdt"
                 / "ch16_per_subject_sdt.csv")
ci = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch22-classification-image"
                 / "ch22_per_subject_ci.csv")
sp = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch24-spectral-logistic"
                 / "ch24_per_subject_spectral.csv")
sl = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
                 / "ch25_per_subject_slopes.csv")
b5 = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
                 / "ch25_per_subject_bin5.csv")
for d in (dp, ci, sp, sl, b5):
    d["subject"] = d["subject"].astype(str)


# ============================================================================
# Significance flags
# ============================================================================

flags = dp[["subject", "site", "block_type", "d_prime", "p_perm_dprime",
            "hit_rate", "fa_rate", "accuracy", "criterion"]].copy()
flags = flags.merge(ci[["subject", "block_type", "cos_ci_true",
                         "p_perm_two", "p_perm_pos"]]
                      .rename(columns = {"p_perm_two": "ci_p_two",
                                         "p_perm_pos": "ci_p_pos"}),
                    on = ["subject", "block_type"], how = "left")
flags = flags.merge(sp[["subject", "block_type", "cos_w_true",
                         "p_perm_two", "p_perm_pos"]]
                      .rename(columns = {"p_perm_two": "spec_p_two",
                                         "p_perm_pos": "spec_p_pos"}),
                    on = ["subject", "block_type"], how = "left")
flags = flags.merge(sl[["subject", "block_type", "beta_truetarget"]],
                    on = ["subject", "block_type"], how = "left")

flags["sig_dprime"] = (flags["p_perm_dprime"] < ALPHA) & (flags["d_prime"] > 0)
flags["sig_ci"]     = (flags["ci_p_pos"]      < ALPHA)
flags["sig_spec"]   = (flags["spec_p_pos"]    < ALPHA)
flags["n_sig"]      = (flags[["sig_dprime", "sig_ci", "sig_spec"]]
                       .astype(int).sum(axis = 1))
flags["any_sig"]    = flags["n_sig"] > 0
flags.to_csv(OUT_PATH / "ch28_significance_flags.csv", index = False)

print("Counts of any_sig per block:", flush = True)
print(flags.groupby("block_type")["any_sig"].sum().to_string(), flush = True)
print("\nCounts of subject-blocks significant on each test:", flush = True)
print(flags.groupby("block_type")[["sig_dprime", "sig_ci", "sig_spec"]]
            .sum().to_string(), flush = True)


# Table of significant rows
sig_table = flags[flags["any_sig"]].copy()
sig_table = sig_table[["site", "subject", "block_type", "d_prime",
                       "p_perm_dprime", "cos_ci_true", "ci_p_pos",
                       "cos_w_true", "spec_p_pos", "beta_truetarget",
                       "hit_rate", "fa_rate", "accuracy"]]
sig_table = sig_table.sort_values(["block_type", "subject"])
sig_table.to_csv(OUT_PATH / "ch28_significant_table.csv", index = False)
print(f"\nSignificant subject-blocks ({len(sig_table)} rows):", flush = True)
print(sig_table.round(3).to_string(index = False), flush = True)


# Subjects significant in *both* blocks
both_blocks = (flags[flags["any_sig"]]
               .groupby("subject")["block_type"].nunique())
stable_subjects = both_blocks[both_blocks == 2].index.tolist()
print(f"\nSubjects significant in BOTH blocks: "
      f"{len(stable_subjects)}  -> {stable_subjects}", flush = True)


# Stably aligned (all three positive in a block)
flags["stably_aligned"] = ((flags["cos_ci_true"]    > 0) &
                           (flags["cos_w_true"]     > 0) &
                           (flags["beta_truetarget"] > 0))
stab = flags[flags["stably_aligned"]].copy()
stab.to_csv(OUT_PATH / "ch28_stably_aligned.csv", index = False)
print(f"\nStably-aligned subject-blocks (all 3 metrics > 0): {len(stab)}",
      flush = True)
print(stab.groupby("block_type")["stably_aligned"].sum().to_string(),
      flush = True)


# ============================================================================
# Questionnaire scores for the significant subjects
# ============================================================================

quest = load_questionnaires()
quest["subject"] = quest["subject"].astype(str)
sig_quest = sig_table.merge(
    quest[["subject", "BAIS_V", "BAIS_C", "VHQ", "TAS", "LSHS", "DES",
           "FSS", "SSS_mean"]], on = "subject", how = "left")
sig_quest.to_csv(OUT_PATH / "ch28_significant_with_quest.csv", index = False)
print("\nQuestionnaire scores for significant subjects:", flush = True)
print(sig_quest[["site", "subject", "block_type",
                  "BAIS_V", "BAIS_C", "TAS", "LSHS"]]
        .round(2).to_string(index = False), flush = True)


# ============================================================================
# Plots
# ============================================================================

# Venn-style overlap counts (drawn as bar chart)
fig, axes = plt.subplots(1, 2, figsize = (12, 5))
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    f = flags[flags["block_type"] == blk]
    cats = ["sig_dprime", "sig_ci", "sig_spec"]
    counts = [int(f[c].sum()) for c in cats]
    ax.bar(["d' (Ch.16)", "CI (Ch.22)", "spec (Ch.24)"], counts,
           color = ["#1f77b4", "#2ca02c", "#d62728"], edgecolor = "k")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.05, str(c), ha = "center")
    ax.set_ylabel("# individually significant subjects (p<0.05)")
    ax.set_title(f"{blk}  (N=26)")
    ax.set_ylim(0, max(counts + [3]) + 1)
plt.suptitle("Per-subject permutation-significant counts by metric",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch28_signif_overlap_venn.png", dpi = 130)
plt.close()


# Heat-strip table of the significant rows
if len(sig_table):
    fig, ax = plt.subplots(figsize = (12, 0.4 * len(sig_table) + 2))
    cell_cols = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget",
                 "hit_rate", "fa_rate", "accuracy"]
    M = sig_table[cell_cols].values.astype(float)
    Mn = M / (np.nanmax(np.abs(M), axis = 0) + 1e-9)
    im = ax.imshow(Mn, aspect = "auto", cmap = "RdBu_r", vmin = -1, vmax = 1)
    ax.set_yticks(np.arange(len(sig_table)))
    ax.set_yticklabels([f"{r['site'][0]} {r['subject'][-3:]} {r['block_type'][:4]}"
                        for _, r in sig_table.iterrows()], fontsize = 8)
    ax.set_xticks(np.arange(len(cell_cols)))
    ax.set_xticklabels(cell_cols, rotation = 45, ha = "right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha = "center", va = "center",
                    fontsize = 7, color = "k")
    plt.colorbar(im, ax = ax, fraction = 0.025,
                 label = "value / col-max(|.|)")
    plt.title("Significant subjects: per-metric values")
    plt.tight_layout()
    plt.savefig(OUT_PATH / "ch28_significant_metric_table.png", dpi = 130)
    plt.close()

print(f"\nChapter 28 done.  Outputs: {OUT_PATH}", flush = True)
