"""Chapter 29.  Questionnaire correlates of within-subject alignment.

Per-subject (per block) we have:
    Performance:   d', cos_ci_true, cos_w_true, beta_truetarget,
                   accuracy, hit_rate, fa_rate, criterion
    Questionnaire: BAIS_V, BAIS_C, VHQ, TAS, LSHS, DES, FSS, SSS_mean

We compute Spearman rho between each questionnaire and each performance
metric, separately per block, and report uncorrected p-values plus an
indication of which survive a simple Bonferroni correction.

For the BAIS scales (the user's primary interest) we additionally:
    - Plot BAIS_V vs each of the four within-subject signal metrics.
    - Run a "high BAIS_V vs low BAIS_V" median split test.

Outputs
-------
outputs/book3/ch29-questionnaires/
    ch29_per_subject_full.csv
    ch29_spearman_full.csv
    ch29_bais_v_scatter.png
    ch29_bais_c_scatter.png
    ch29_high_low_bais_v_test.csv
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book3_helpers import load_questionnaires

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch29-questionnaires"
OUT_PATH.mkdir(parents = True, exist_ok = True)

QUEST_COLS = ["BAIS_V", "BAIS_C", "VHQ", "TAS", "LSHS", "DES", "FSS",
              "SSS_mean"]
PERF_COLS  = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget",
              "accuracy", "hit_rate", "fa_rate", "criterion"]


# ============================================================================
# Pull data
# ============================================================================

dp = pd.read_csv(CUR_PATH / "outputs" / "book2" / "ch16-sdt"
                 / "ch16_per_subject_sdt.csv")
ci = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch22-classification-image"
                 / "ch22_per_subject_ci.csv")
sp = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch24-spectral-logistic"
                 / "ch24_per_subject_spectral.csv")
sl = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
                 / "ch25_per_subject_slopes.csv")
for d in (dp, ci, sp, sl):
    d["subject"] = d["subject"].astype(str)

merged = dp[["subject", "site", "block_type", "d_prime", "hit_rate",
              "fa_rate", "accuracy", "criterion"]] \
        .merge(ci[["subject", "block_type", "cos_ci_true"]],
               on = ["subject", "block_type"]) \
        .merge(sp[["subject", "block_type", "cos_w_true"]],
               on = ["subject", "block_type"]) \
        .merge(sl[["subject", "block_type", "beta_truetarget"]],
               on = ["subject", "block_type"])

quest = load_questionnaires()
quest["subject"] = quest["subject"].astype(str)

merged = merged.merge(quest[["subject"] + QUEST_COLS], on = "subject",
                      how = "left")
merged.to_csv(OUT_PATH / "ch29_per_subject_full.csv", index = False)
print(f"Merged rows: {len(merged)}", flush = True)


# ============================================================================
# Spearman correlations
# ============================================================================

rows = []
for blk, g in merged.groupby("block_type"):
    for q in QUEST_COLS:
        for p in PERF_COLS:
            x = g[q].values; y = g[p].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 8:
                continue
            rho, pv = spearmanr(x[ok], y[ok])
            rows.append(dict(block_type = blk, quest = q, perf = p,
                              n = int(ok.sum()),
                              rho = float(rho), p = float(pv)))
spear = pd.DataFrame(rows)
n_tests = len(spear)
spear["p_bonf"] = (spear["p"] * n_tests).clip(upper = 1.0)
spear.to_csv(OUT_PATH / "ch29_spearman_full.csv", index = False)

print(f"\nTotal Spearman tests: {n_tests}; Bonf threshold: {0.05 / n_tests:.5f}",
      flush = True)
print("\nUncorrected p<0.05 correlations:", flush = True)
print(spear[spear["p"] < 0.05]
      .sort_values("p")
      .round(4).to_string(index = False), flush = True)
print("\nBonferroni-significant correlations:", flush = True)
print(spear[spear["p_bonf"] < 0.05]
      .sort_values("p")
      .round(4).to_string(index = False), flush = True)


# ============================================================================
# BAIS scatter plots
# ============================================================================

def bais_scatter(quest_col, fname):
    fig, axes = plt.subplots(2, 4, figsize = (16, 8))
    plot_metrics = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget"]
    site_color = {"Oberlin": "#2ca02c", "UChicago": "#9467bd"}
    for i, blk in enumerate(("full_sentence", "imagined_sentence")):
        g = merged[merged["block_type"] == blk]
        for j, m in enumerate(plot_metrics):
            ax = axes[i, j]
            for site, gg in g.groupby("site"):
                ax.scatter(gg[quest_col], gg[m], color = site_color[site],
                           edgecolor = "k", s = 55, alpha = 0.85, label = site)
            x = g[quest_col].values; y = g[m].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() > 3:
                rho, pv = spearmanr(x[ok], y[ok])
                ax.set_title(f"{blk}\n{m}   rho={rho:+.2f} p={pv:.3f}",
                             fontsize = 9)
                # OLS line for visual
                slope, intc = np.polyfit(x[ok], y[ok], 1)
                xs = np.linspace(x[ok].min(), x[ok].max(), 50)
                ax.plot(xs, slope * xs + intc, color = "k", lw = 1, ls = "--",
                        alpha = 0.7)
            ax.axhline(0, color = "gray", lw = 0.5, ls = ":")
            ax.set_xlabel(quest_col)
    axes[0, 0].legend(fontsize = 8)
    plt.suptitle(f"Per-subject performance vs {quest_col}",
                 fontsize = 12)
    plt.tight_layout()
    plt.savefig(OUT_PATH / fname, dpi = 130)
    plt.close()


bais_scatter("BAIS_V", "ch29_bais_v_scatter.png")
bais_scatter("BAIS_C", "ch29_bais_c_scatter.png")
bais_scatter("TAS",    "ch29_tas_scatter.png")
bais_scatter("LSHS",   "ch29_lshs_scatter.png")


# ============================================================================
# Median-split test on BAIS_V
# ============================================================================

med_split_rows = []
for blk, g in merged.groupby("block_type"):
    g = g.dropna(subset = ["BAIS_V"])
    cutoff = g["BAIS_V"].median()
    high = g[g["BAIS_V"] >  cutoff]
    low  = g[g["BAIS_V"] <= cutoff]
    for m in ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget"]:
        try:
            stat, p = mannwhitneyu(high[m].dropna(), low[m].dropna(),
                                   alternative = "greater")
        except ValueError:
            stat, p = (np.nan, np.nan)
        med_split_rows.append(dict(block_type = blk, metric = m,
                                   bais_v_cut = float(cutoff),
                                   mean_high = float(high[m].mean()),
                                   mean_low  = float(low[m].mean()),
                                   n_high = len(high), n_low = len(low),
                                   U = stat, p_one_sided = p))
ms_df = pd.DataFrame(med_split_rows)
ms_df.to_csv(OUT_PATH / "ch29_high_low_bais_v_test.csv", index = False)
print("\nBAIS_V median-split (one-sided: high > low):", flush = True)
print(ms_df.round(4).to_string(index = False), flush = True)

print(f"\nChapter 29 done.  Outputs: {OUT_PATH}", flush = True)
