"""Chapter 26.  Block-order effects and site-level split.

Two questions:
    (A) Block order: does d', alignment, and other per-subject metrics
        differ between trials run as Block 1 vs Block 2?  (Practice / fatigue.)
    (B) Site: do the per-subject metrics from chapters 16, 22, 24, 25 differ
        between Oberlin (N=16) and UChicago (N=10)?

For each per-subject scalar (d', cos_ci_true, cos_w_true, beta_truetarget)
we:
    1. Plot the per-subject distribution faceted by block-order and by site.
    2. Run Wilcoxon paired (block-order: same subject's b1 vs b2 estimates,
       same block_type) and Mann-Whitney unpaired (site).

Outputs
-------
outputs/book3/ch26-block-order-and-site/
    ch26_metrics_long.csv
    ch26_block_order_tests.csv
    ch26_site_tests.csv
    ch26_block_order.png
    ch26_site_split.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book3_helpers import load_block_order

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch26-block-order-and-site"
OUT_PATH.mkdir(parents = True, exist_ok = True)


# ============================================================================
# Pull the four per-subject metrics
# ============================================================================

def _read(path, val_col, new_name):
    df = pd.read_csv(path)
    df["subject"] = df["subject"].astype(str)
    return df[["subject", "block_type", val_col]].rename(columns = {val_col: new_name})

dp  = _read(CUR_PATH / "outputs" / "book2" / "ch16-sdt"
            / "ch16_per_subject_sdt.csv", "d_prime", "d_prime")
ci  = _read(CUR_PATH / "outputs" / "book3" / "ch22-classification-image"
            / "ch22_per_subject_ci.csv", "cos_ci_true", "cos_ci_true")
sp  = _read(CUR_PATH / "outputs" / "book3" / "ch24-spectral-logistic"
            / "ch24_per_subject_spectral.csv", "cos_w_true", "cos_w_true")
sl  = _read(CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
            / "ch25_per_subject_slopes.csv", "beta_truetarget",
            "beta_truetarget")

bo = load_block_order()
bo["subject"] = bo["subject"].astype(str)

merged = bo.merge(dp, on = ["subject", "block_type"], how = "left") \
           .merge(ci, on = ["subject", "block_type"], how = "left") \
           .merge(sp, on = ["subject", "block_type"], how = "left") \
           .merge(sl, on = ["subject", "block_type"], how = "left")
merged.to_csv(OUT_PATH / "ch26_metrics_long.csv", index = False)
print(f"Merged rows: {len(merged)}", flush = True)

METRICS = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget"]


# ============================================================================
# (A) Block order test
# ============================================================================

bo_rows = []
for blk in ("full_sentence", "imagined_sentence"):
    sub = merged[merged["block_type"] == blk]
    for m in METRICS:
        b1 = sub[sub["block_order_index"] == 1][m].dropna().values
        b2 = sub[sub["block_order_index"] == 2][m].dropna().values
        try:
            stat, p = mannwhitneyu(b1, b2, alternative = "two-sided")
        except ValueError:
            stat, p = (np.nan, np.nan)
        bo_rows.append(dict(block_type = blk, metric = m,
                            mean_b1 = float(np.mean(b1)) if len(b1) else np.nan,
                            mean_b2 = float(np.mean(b2)) if len(b2) else np.nan,
                            n_b1 = len(b1), n_b2 = len(b2),
                            U = stat, p = p))
bo_df = pd.DataFrame(bo_rows)
bo_df.to_csv(OUT_PATH / "ch26_block_order_tests.csv", index = False)
print("\nBlock-order Mann-Whitney (b1 vs b2 within block_type):", flush = True)
print(bo_df.to_string(index = False), flush = True)


# Within-subject paired: each subject contributes one b1 d' and one b2 d'
# (across the two block_types).  This tests whether the block run *first*
# is better/worse than the block run *second* irrespective of which scheme.
paired_rows = []
for m in METRICS:
    pivot = merged.pivot_table(index = "subject", columns = "block_order_index",
                               values = m, aggfunc = "first")
    pivot = pivot.dropna()
    if len(pivot) < 5:
        continue
    diff = pivot[2] - pivot[1]
    stat, p = wilcoxon(diff)
    paired_rows.append(dict(metric = m, n = len(pivot),
                            mean_b1 = float(pivot[1].mean()),
                            mean_b2 = float(pivot[2].mean()),
                            mean_diff_b2_minus_b1 = float(diff.mean()),
                            W = float(stat), p = float(p)))
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUT_PATH / "ch26_block_order_paired.csv", index = False)
print("\nBlock-order paired Wilcoxon (b2 - b1, irrespective of block_type):",
      flush = True)
print(paired_df.to_string(index = False), flush = True)


# ============================================================================
# (B) Site test
# ============================================================================

site_rows = []
for blk in ("full_sentence", "imagined_sentence"):
    sub = merged[merged["block_type"] == blk]
    for m in METRICS:
        ob = sub[sub["site"] == "Oberlin"][m].dropna().values
        uc = sub[sub["site"] == "UChicago"][m].dropna().values
        try:
            stat, p = mannwhitneyu(ob, uc, alternative = "two-sided")
        except ValueError:
            stat, p = (np.nan, np.nan)
        site_rows.append(dict(block_type = blk, metric = m,
                              mean_oberlin = float(np.mean(ob)) if len(ob) else np.nan,
                              mean_uchicago = float(np.mean(uc)) if len(uc) else np.nan,
                              n_oberlin = len(ob), n_uchicago = len(uc),
                              U = stat, p = p))
site_df = pd.DataFrame(site_rows)
site_df.to_csv(OUT_PATH / "ch26_site_tests.csv", index = False)
print("\nSite Mann-Whitney (Oberlin vs UChicago):", flush = True)
print(site_df.to_string(index = False), flush = True)


# ============================================================================
# Plots
# ============================================================================

# Block-order strip + mean
fig, axes = plt.subplots(2, 4, figsize = (16, 8), sharex = "col")
rng = np.random.default_rng(0)
for i, blk in enumerate(("full_sentence", "imagined_sentence")):
    sub = merged[merged["block_type"] == blk]
    for j, m in enumerate(METRICS):
        ax = axes[i, j]
        for k, order in enumerate([1, 2]):
            v = sub[sub["block_order_index"] == order][m].dropna().values
            x = np.full_like(v, k, dtype = float) + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, color = ["#1f77b4", "#d62728"][k], edgecolor = "k",
                       s = 45, alpha = 0.85)
            ax.hlines(v.mean(), k - 0.25, k + 0.25, color = "k", lw = 2)
        ax.axhline(0, color = "gray", lw = 0.6, ls = "--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Block 1", "Block 2"])
        ax.set_title(f"{blk}\n{m}", fontsize = 9)
plt.suptitle("Per-subject metrics by block-order index, within block_type",
             fontsize = 12)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch26_block_order.png", dpi = 130)
plt.close()

# Site strip plot
fig, axes = plt.subplots(2, 4, figsize = (16, 8), sharex = "col")
for i, blk in enumerate(("full_sentence", "imagined_sentence")):
    sub = merged[merged["block_type"] == blk]
    for j, m in enumerate(METRICS):
        ax = axes[i, j]
        for k, site in enumerate(["Oberlin", "UChicago"]):
            v = sub[sub["site"] == site][m].dropna().values
            x = np.full_like(v, k, dtype = float) + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, color = ["#2ca02c", "#9467bd"][k], edgecolor = "k",
                       s = 45, alpha = 0.85)
            if len(v):
                ax.hlines(v.mean(), k - 0.25, k + 0.25, color = "k", lw = 2)
        ax.axhline(0, color = "gray", lw = 0.6, ls = "--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Oberlin\n(N=16)", "UChicago\n(N=10)"])
        ax.set_title(f"{blk}\n{m}", fontsize = 9)
plt.suptitle("Per-subject metrics by site, within block_type", fontsize = 12)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch26_site_split.png", dpi = 130)
plt.close()

print(f"\nChapter 26 done.  Outputs: {OUT_PATH}", flush = True)
