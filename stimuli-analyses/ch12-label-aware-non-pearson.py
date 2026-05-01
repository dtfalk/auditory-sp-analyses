"""Chapter 12. Label-Aware Group Structure Under Alternative Metrics.

For each metric loaded from ch11, compute:
  - within-target similarity distribution            (tt)
  - within-distractor similarity distribution        (dd)
  - between-group target-distractor distribution     (td)
  - effect sizes:
      Cohen's d (tt vs dd)        -> within-group separation
      Cohen's d (tt vs td)        -> targets vs cross
      Cohen's d (dd vs td)        -> distractors vs cross
  - Welch t-test p-values for the same comparisons
  - Comparison against the Pearson reference

Outputs
-------
outputs/non_pearson/ch12-label-aware/
    ch12_summary.csv              - one row per metric x comparison
    ch12_per_metric_distributions.csv - long format: metric, group, similarity
    ch12_distributions_<metric>.png   - per-metric overlapping histograms
    ch12_cohesion_comparison.png      - bar chart: tt-dd separation vs Pearson
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def cohens_d(a, b):
    """Cohen's d for two independent samples (pooled SD).

    Args:
        a: 1-D array.
        b: 1-D array.

    Returns:
        Float effect size (mean(a) - mean(b)) / pooled_sd.
    """
    n_a   = len(a)
    n_b   = len(b)
    var_a = np.var(a, ddof = 1)
    var_b = np.var(b, ddof = 1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd


def fmt_p(p):
    """Format a p-value compactly."""
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4f}"


def split_by_label(sim_matrix, labels):
    """Extract tt, dd, td similarity vectors from a labeled similarity matrix.

    Args:
        sim_matrix: (n, n) symmetric similarity matrix.
        labels:     (n,) array of 'target' / 'distractor'.

    Returns:
        Tuple (tt, dd, td) of 1-D arrays.
    """
    is_t  = labels == "target"
    is_d  = labels == "distractor"
    t_idx = np.where(is_t)[0]
    d_idx = np.where(is_d)[0]

    tt_block = sim_matrix[np.ix_(t_idx, t_idx)]
    dd_block = sim_matrix[np.ix_(d_idx, d_idx)]
    td_block = sim_matrix[np.ix_(t_idx, d_idx)]

    tt = tt_block[np.triu_indices_from(tt_block, k = 1)]
    dd = dd_block[np.triu_indices_from(dd_block, k = 1)]
    td = td_block.ravel()
    return tt, dd, td


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path    = Path(__file__).resolve().parent
    in_path     = cur_path / "outputs" / "non_pearson" / "ch11-metrics" / "similarity_matrices.npz"
    output_path = cur_path / "outputs" / "non_pearson" / "ch12-label-aware"
    output_path.mkdir(parents = True, exist_ok = True)

    assert in_path.exists(), f"Run ch11 first; missing {in_path}"
    data        = np.load(in_path, allow_pickle = False)
    labels      = data["labels"]
    metric_names = ["pearson_raw", "lagged_xcorr_max", "envelope_corr",
                    "log_spectrum_corr", "mfcc_cosine"]

    summary_rows = []
    long_rows    = []

    for metric in metric_names:
        sim       = data[metric]
        tt, dd, td = split_by_label(sim, labels)

        d_tt_dd   = cohens_d(tt, dd)
        d_tt_td   = cohens_d(tt, td)
        d_dd_td   = cohens_d(dd, td)
        _, p_tt_dd = ttest_ind(tt, dd, equal_var = False)
        _, p_tt_td = ttest_ind(tt, td, equal_var = False)
        _, p_dd_td = ttest_ind(dd, td, equal_var = False)

        summary_rows.append({
            "metric":   metric,
            "mean_tt":  float(tt.mean()),
            "mean_dd":  float(dd.mean()),
            "mean_td":  float(td.mean()),
            "std_tt":   float(tt.std(ddof = 1)),
            "std_dd":   float(dd.std(ddof = 1)),
            "std_td":   float(td.std(ddof = 1)),
            "n_tt":     int(len(tt)),
            "n_dd":     int(len(dd)),
            "n_td":     int(len(td)),
            "cohens_d_tt_vs_dd": float(d_tt_dd),
            "cohens_d_tt_vs_td": float(d_tt_td),
            "cohens_d_dd_vs_td": float(d_dd_td),
            "welch_p_tt_vs_dd":  float(p_tt_dd),
            "welch_p_tt_vs_td":  float(p_tt_td),
            "welch_p_dd_vs_td":  float(p_dd_td),
        })

        for grp_name, vals in [("tt", tt), ("dd", dd), ("td", td)]:
            for v in vals:
                long_rows.append({"metric": metric, "group": grp_name, "similarity": float(v)})

        print(f"{metric:20s}  mean_tt={tt.mean():+.4f}  mean_dd={dd.mean():+.4f}  "
              f"mean_td={td.mean():+.4f}  d(tt-dd)={d_tt_dd:+.3f}  "
              f"d(tt-td)={d_tt_td:+.3f}  p(tt-dd)={fmt_p(p_tt_dd)}", flush = True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / "ch12_summary.csv", index = False)
    pd.DataFrame(long_rows).to_csv(output_path / "ch12_per_metric_distributions.csv",
                                   index = False)

    # -----------------------------------------------------------------------
    # Plot 1: per-metric overlapping histograms.
    # -----------------------------------------------------------------------
    for metric in metric_names:
        sim       = data[metric]
        tt, dd, td = split_by_label(sim, labels)

        fig, ax = plt.subplots(figsize = (8, 5))
        bins = np.linspace(min(tt.min(), dd.min(), td.min()),
                           max(tt.max(), dd.max(), td.max()), 60)
        ax.hist(td, bins = bins, color = "lightgray",  edgecolor = "black",
                alpha = 0.85, label = f"target-distractor (n={len(td)})")
        ax.hist(tt, bins = bins, color = "tomato",     edgecolor = "black",
                alpha = 0.55, label = f"target-target   (n={len(tt)})")
        ax.hist(dd, bins = bins, color = "steelblue",  edgecolor = "black",
                alpha = 0.55, label = f"distr-distr     (n={len(dd)})")
        ax.axvline(tt.mean(), color = "tomato",    linewidth = 2)
        ax.axvline(dd.mean(), color = "steelblue", linewidth = 2)
        ax.axvline(td.mean(), color = "black",     linewidth = 2, linestyle = "--")
        ax.set_title(f"{metric}: pairwise similarity by group")
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize = 8)
        fig.tight_layout()
        fig.savefig(output_path / f"ch12_distributions_{metric}.png", dpi = 150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: separation comparison across metrics (Cohen's d).
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (10, 5))
    x      = np.arange(len(metric_names))
    width  = 0.28
    d1     = [r["cohens_d_tt_vs_dd"] for r in summary_rows]
    d2     = [r["cohens_d_tt_vs_td"] for r in summary_rows]
    d3     = [r["cohens_d_dd_vs_td"] for r in summary_rows]
    ax.bar(x - width, d1, width, color = "tomato",    label = "d(tt vs dd)")
    ax.bar(x,         d2, width, color = "darkorange", label = "d(tt vs td)")
    ax.bar(x + width, d3, width, color = "steelblue", label = "d(dd vs td)")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Group separation by metric (positive = first group higher similarity)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "ch12_cohesion_comparison.png", dpi = 150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: separation relative to Pearson reference.
    # -----------------------------------------------------------------------
    pearson_d_tt_dd = summary_df.set_index("metric").loc["pearson_raw", "cohens_d_tt_vs_dd"]
    rel = [r["cohens_d_tt_vs_dd"] / pearson_d_tt_dd if pearson_d_tt_dd != 0 else np.nan
           for r in summary_rows]
    fig, ax = plt.subplots(figsize = (8, 4.5))
    ax.bar(metric_names, rel, color = "darkorange", edgecolor = "black")
    ax.axhline(1.0, color = "black", linewidth = 0.8, linestyle = "--",
               label = "Pearson reference")
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("d(tt vs dd) / d_pearson(tt vs dd)")
    ax.set_title("Within-group separation relative to Pearson reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "ch12_relative_to_pearson.png", dpi = 150)
    plt.close(fig)

    print(f"Chapter 12 done.  Outputs saved to: {output_path}", flush = True)


if __name__ == "__main__":
    main()
