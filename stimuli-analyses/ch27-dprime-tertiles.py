"""Chapter 27.  d' tertiles -- signed and absolute -- and what predicts them.

For each block, sort subjects by their Ch.16 d' and split into 3 tertiles
(low / medium / high).  Then repeat for |d'|.

For each tertile we report:
    n, mean d', mean cos_ci_true (Ch.22), mean cos_w_true (Ch.24),
    mean beta_truetarget (Ch.25), mean response_rate, hit, FA, accuracy.

We then test whether the tertiles differ (Kruskal-Wallis) on each metric
and run a per-tertile pairwise comparison (Mann-Whitney) for the metric
of primary interest, cos_w_true.

Outputs
-------
outputs/book3/ch27-dprime-tertiles/
    ch27_per_subject_tertiles.csv
    ch27_signed_tertile_summary.csv
    ch27_abs_tertile_summary.csv
    ch27_signed_tertile_kruskal.csv
    ch27_abs_tertile_kruskal.csv
    ch27_signed_tertile_box.png
    ch27_abs_tertile_box.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch27-dprime-tertiles"
OUT_PATH.mkdir(parents = True, exist_ok = True)


# ============================================================================
# Pull metrics
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
            / "ch25_per_subject_slopes.csv", "beta_truetarget", "beta_truetarget")
sdt_full = pd.read_csv(CUR_PATH / "outputs" / "book2" / "ch16-sdt"
                       / "ch16_per_subject_sdt.csv")
sdt_full["subject"] = sdt_full["subject"].astype(str)

merged = dp.merge(ci, on = ["subject", "block_type"]) \
           .merge(sp, on = ["subject", "block_type"]) \
           .merge(sl, on = ["subject", "block_type"]) \
           .merge(sdt_full[["subject", "block_type", "site", "hit_rate",
                             "fa_rate", "accuracy", "criterion"]],
                  on = ["subject", "block_type"])
merged["abs_d_prime"] = merged["d_prime"].abs()


# ============================================================================
# Tertile assignment
# ============================================================================

def tertile(s):
    """Return labels low/med/high using qcut on series s."""
    return pd.qcut(s, 3, labels = ["low", "med", "high"])

merged["signed_tertile"] = (merged.groupby("block_type")["d_prime"]
                                  .transform(tertile))
merged["abs_tertile"]    = (merged.groupby("block_type")["abs_d_prime"]
                                  .transform(tertile))

merged.to_csv(OUT_PATH / "ch27_per_subject_tertiles.csv", index = False)


# ============================================================================
# Tertile summary + Kruskal
# ============================================================================

ANALYSIS_COLS = ["d_prime", "abs_d_prime", "cos_ci_true", "cos_w_true",
                 "beta_truetarget", "hit_rate", "fa_rate", "accuracy",
                 "criterion"]

def tertile_summary(df, tertile_col):
    summ_rows = []
    krus_rows = []
    for blk, g in df.groupby("block_type"):
        for col in ANALYSIS_COLS:
            for tert in ["low", "med", "high"]:
                v = g.loc[g[tertile_col] == tert, col].dropna().values
                summ_rows.append(dict(block_type = blk, tertile = tert,
                                      metric = col,
                                      n = len(v),
                                      mean = float(np.mean(v)) if len(v) else np.nan,
                                      median = float(np.median(v)) if len(v) else np.nan,
                                      std    = float(np.std(v))    if len(v) else np.nan))
            groups = [g.loc[g[tertile_col] == t, col].dropna().values
                      for t in ["low", "med", "high"]]
            try:
                stat, p = kruskal(*groups)
            except ValueError:
                stat, p = (np.nan, np.nan)
            krus_rows.append(dict(block_type = blk, metric = col,
                                  H = stat, p = p))
    return pd.DataFrame(summ_rows), pd.DataFrame(krus_rows)


for label, col in [("signed", "signed_tertile"), ("abs", "abs_tertile")]:
    summ, krus = tertile_summary(merged, col)
    summ.to_csv(OUT_PATH / f"ch27_{label}_tertile_summary.csv", index = False)
    krus.to_csv(OUT_PATH / f"ch27_{label}_tertile_kruskal.csv", index = False)
    print(f"\n=== {label} tertile (split by {'d_prime' if label=='signed' else '|d_prime|'}) ===",
          flush = True)
    print("Means by tertile (block × metric):", flush = True)
    pivot = summ.pivot_table(index = ["block_type", "tertile"], columns = "metric",
                             values = "mean")
    pivot = pivot.reindex(["low", "med", "high"], level = 1)
    print(pivot.round(3).to_string(), flush = True)
    print(f"\n{label}-tertile Kruskal-Wallis (across 3 tertiles):", flush = True)
    print(krus.to_string(index = False), flush = True)

    sig = krus[krus["p"] < 0.10]
    if len(sig):
        print(f"\nKruskal p<0.10 ({label}):", flush = True)
        print(sig.to_string(index = False), flush = True)


# ============================================================================
# Plots
# ============================================================================

def tertile_box(tertile_col, fname, label):
    fig, axes = plt.subplots(2, 4, figsize = (16, 8))
    plot_metrics = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget"]
    rng = np.random.default_rng(0)
    for i, blk in enumerate(("full_sentence", "imagined_sentence")):
        g = merged[merged["block_type"] == blk]
        for j, m in enumerate(plot_metrics):
            ax = axes[i, j]
            data = [g.loc[g[tertile_col] == t, m].dropna().values
                    for t in ["low", "med", "high"]]
            bp = ax.boxplot(data, positions = [0, 1, 2], widths = 0.55,
                            patch_artist = True, showmeans = True,
                            meanprops = dict(marker = "D", markerfacecolor = "white",
                                             markeredgecolor = "k"))
            for patch, c in zip(bp["boxes"], ["#fdae61", "#ffffbf", "#abdda4"]):
                patch.set_facecolor(c)
            for k, v in enumerate(data):
                jit = rng.uniform(-0.12, 0.12, len(v))
                ax.scatter(np.full_like(v, k, dtype = float) + jit, v,
                           color = "k", s = 18, alpha = 0.7)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "med", "high"])
            ax.axhline(0, color = "gray", lw = 0.6, ls = "--")
            ax.set_title(f"{blk}\n{m}", fontsize = 9)
    plt.suptitle(f"Per-subject metrics by {label} d' tertile, within block",
                 fontsize = 12)
    plt.tight_layout()
    plt.savefig(OUT_PATH / fname, dpi = 130)
    plt.close()


tertile_box("signed_tertile", "ch27_signed_tertile_box.png", "signed")
tertile_box("abs_tertile",    "ch27_abs_tertile_box.png",    "absolute")

print(f"\nChapter 27 done.  Outputs: {OUT_PATH}", flush = True)
