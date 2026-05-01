"""Chapter 16.  Signal-detection alignment & response bias.

Question
--------
For each subject and each block, how well do their responses align with the
experimenter target/distractor labels, after separating sensitivity (d') from
response bias (criterion C)?  And does the group as a whole show evidence of
sensitivity beyond chance?

Per-subject metrics (per block):
    n_trials, hit_rate, fa_rate, d_prime, criterion, accuracy, auc
A label-permutation null gives a per-subject p-value on d_prime; a group-level
test uses Wilcoxon signed-rank against zero on d_prime across subjects.

Outputs
-------
outputs/book2/ch16-sdt/
    ch16_per_subject_sdt.csv
    ch16_group_summary.csv
    ch16_dprime_forest.png
    ch16_dprime_vs_criterion.png
    ch16_roc_per_subject.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats          import norm, wilcoxon, ttest_1samp
from sklearn.metrics      import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import load_all_trials

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book2" / "ch16-sdt"
OUT_PATH.mkdir(parents = True, exist_ok = True)


# ============================================================================
# SDT helpers
# ============================================================================

def sdt_metrics(true_target, called_target):
    """Return dict of hit_rate, fa_rate, d_prime, criterion, accuracy, auc."""
    true_target   = np.asarray(true_target,   dtype = int)
    called_target = np.asarray(called_target, dtype = int)
    n_t = (true_target == 1).sum()
    n_d = (true_target == 0).sum()
    hits = ((true_target == 1) & (called_target == 1)).sum()
    fas  = ((true_target == 0) & (called_target == 1)).sum()
    # log-linear correction (Hautus 1995): add 0.5 to hit/fa counts, 1.0 to N
    h = (hits + 0.5) / (n_t + 1.0)
    f = (fas  + 0.5) / (n_d + 1.0)
    z_h = norm.ppf(h)
    z_f = norm.ppf(f)
    d_prime   = z_h - z_f
    criterion = -0.5 * (z_h + z_f)
    acc = (called_target == true_target).mean()
    # AUC needs at least one of each class
    if n_t > 0 and n_d > 0:
        auc = roc_auc_score(true_target, called_target)
    else:
        auc = np.nan
    return dict(hit_rate = hits / max(n_t, 1),
                fa_rate  = fas  / max(n_d, 1),
                d_prime  = d_prime,
                criterion = criterion,
                accuracy = acc,
                auc = auc,
                n_target = int(n_t),
                n_distractor = int(n_d))


def perm_p_dprime(true_target, called_target, n_perm = 2000, rng = None):
    """Two-sided p that observed |d'| exceeds label-permutation null."""
    rng = rng if rng is not None else np.random.default_rng(0)
    obs = sdt_metrics(true_target, called_target)["d_prime"]
    null = np.empty(n_perm)
    tt = np.asarray(true_target).copy()
    for b in range(n_perm):
        ttp = rng.permutation(tt)
        null[b] = sdt_metrics(ttp, called_target)["d_prime"]
    p = (1 + np.sum(np.abs(null) >= np.abs(obs))) / (n_perm + 1)
    return obs, p, null


# ============================================================================
# Main
# ============================================================================

print("Loading trials ...", flush = True)
trials = load_all_trials()
print(f"  {len(trials)} trials, {trials['subject'].nunique()} subjects, "
      f"{trials['block_type'].nunique()} blocks", flush = True)

print("Per-subject SDT ...", flush = True)
rng  = np.random.default_rng(0)
rows = []
for (site, subj, block), sub in trials.groupby(["site", "subject", "block_type"]):
    m = sdt_metrics(sub["true_target"].to_numpy(), sub["called_target"].to_numpy())
    obs, p_perm, _ = perm_p_dprime(sub["true_target"].to_numpy(),
                                   sub["called_target"].to_numpy(),
                                   n_perm = 1000, rng = rng)
    rows.append(dict(site = site, subject = subj, block_type = block,
                     **m, p_perm_dprime = p_perm))
per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch16_per_subject_sdt.csv", index = False)
print(per_subj.describe().T[["mean", "std", "min", "max"]], flush = True)


# ============================================================================
# Group-level: Wilcoxon vs zero on d', accuracy vs 0.5
# ============================================================================

group_rows = []
for block, g in per_subj.groupby("block_type"):
    d  = g["d_prime"].to_numpy()
    a  = g["accuracy"].to_numpy()
    c  = g["criterion"].to_numpy()
    w_d, p_d_w = wilcoxon(d, alternative = "greater")
    t_d, p_d_t = ttest_1samp(d, 0.0, alternative = "greater")
    t_c, p_c_t = ttest_1samp(c, 0.0, alternative = "two-sided")
    n_sig = int((g["p_perm_dprime"] < 0.05).sum())
    group_rows.append(dict(block_type = block,
                           n_subjects = len(g),
                           mean_dprime = d.mean(), median_dprime = np.median(d),
                           mean_accuracy = a.mean(), mean_criterion = c.mean(),
                           wilcoxon_dprime_stat = w_d, wilcoxon_dprime_p = p_d_w,
                           ttest_dprime_t = t_d, ttest_dprime_p = p_d_t,
                           ttest_criterion_t = t_c, ttest_criterion_p = p_c_t,
                           n_subj_perm_sig = n_sig))
group = pd.DataFrame(group_rows)
group.to_csv(OUT_PATH / "ch16_group_summary.csv", index = False)
print(group.to_string(index = False), flush = True)


# ============================================================================
# Plots
# ============================================================================

# Forest of d' per (subject, block)
fig, ax = plt.subplots(figsize = (8, max(6, 0.25 * per_subj["subject"].nunique())))
order = (per_subj.groupby("subject")["d_prime"].mean()
         .sort_values().index.tolist())
y_pos = {s: i for i, s in enumerate(order)}
colors = {"full_sentence": "#1f77b4", "imagined_sentence": "#d62728"}
for _, r in per_subj.iterrows():
    ax.scatter(r["d_prime"], y_pos[r["subject"]],
               color = colors[r["block_type"]], s = 30,
               edgecolor = "k", linewidth = 0.4, zorder = 3)
ax.axvline(0, color = "k", lw = 0.8)
ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(order, fontsize = 7)
ax.set_xlabel("d' (label-aligned sensitivity)")
ax.set_title("Per-subject d' by block")
for blk, c in colors.items():
    ax.scatter([], [], color = c, label = blk)
ax.legend(loc = "lower right", fontsize = 8)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch16_dprime_forest.png", dpi = 130)
plt.close()

# d' vs criterion scatter
fig, ax = plt.subplots(figsize = (7, 6))
for blk, g in per_subj.groupby("block_type"):
    ax.scatter(g["criterion"], g["d_prime"], color = colors[blk],
               edgecolor = "k", s = 60, alpha = 0.85, label = blk)
ax.axhline(0, color = "gray", lw = 0.7)
ax.axvline(0, color = "gray", lw = 0.7)
ax.set_xlabel("Criterion C  (negative = liberal / 'target' bias)")
ax.set_ylabel("d'  (label-aligned sensitivity)")
ax.set_title("Sensitivity vs response bias, per subject")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PATH / "ch16_dprime_vs_criterion.png", dpi = 130)
plt.close()

# ROC overlay per subject (binary responses give a single point each, but we
# can stack them as a cloud by subject + show diagonal)
fig, axes = plt.subplots(1, 2, figsize = (12, 5.5), sharex = True, sharey = True)
for ax, blk in zip(axes, ["full_sentence", "imagined_sentence"]):
    g = per_subj[per_subj["block_type"] == blk]
    ax.plot([0, 1], [0, 1], "--", color = "gray", lw = 0.8)
    ax.scatter(g["fa_rate"], g["hit_rate"], color = colors[blk],
               edgecolor = "k", s = 70, alpha = 0.85)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"{blk}: ROC point per subject")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch16_roc_per_subject.png", dpi = 130)
plt.close()

print(f"Chapter 16 done.  Outputs saved to: {OUT_PATH}", flush = True)
