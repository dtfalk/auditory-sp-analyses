"""Chapter 14. Metric Comparison: Which Dimensions Carry the Most Structure?

Synthesizes Part II + Part III by ranking metrics on three orthogonal axes:

  1. Target cohesion             - mean_tt under each metric (already computed in
                                    ch12), normalized as a z-score against the
                                    metric's own off-diagonal distribution to
                                    make different metric scales comparable.
  2. Label separation            - Cohen's d (tt vs dd) under each metric (ch12).
  3. Intrinsic cluster strength  - silhouette score of agglomerative k=2 cluster
                                    (ch13).

Also computes the Mantel-style cross-metric correlation: for each pair of
metrics, the Pearson r between the flattened upper-triangular pairwise
similarity vectors (44850 pairs).  Tells us which metrics "see" similar
pair-relations and which carve up the space differently.

Outputs
-------
outputs/non_pearson/ch14-comparison/
    ch14_summary_ranked.csv             - metric rankings on each axis
    ch14_cross_metric_mantel.csv        - 5x5 Mantel-style r matrix
    ch14_cross_metric_mantel.png        - heatmap
    ch14_axis_rankings.png              - bar chart of the three axis scores
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def upper_tri_flat(mat):
    """Flatten upper triangle (k=1) of a square matrix.

    Args:
        mat: (n, n) array.

    Returns:
        1-D array of length n * (n - 1) / 2.
    """
    return mat[np.triu_indices_from(mat, k = 1)]


def split_tt_off(sim, labels):
    """Return (tt vector, full off-diagonal vector) for cohesion z-score.

    Args:
        sim:    (n, n) similarity matrix.
        labels: (n,) array of group labels.

    Returns:
        Tuple (tt, off) of 1-D arrays.
    """
    is_t  = labels == "target"
    t_idx = np.where(is_t)[0]
    tt_block = sim[np.ix_(t_idx, t_idx)]
    tt = tt_block[np.triu_indices_from(tt_block, k = 1)]
    off = upper_tri_flat(sim)
    return tt, off


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path     = Path(__file__).resolve().parent
    np_root      = cur_path / "outputs" / "non_pearson"
    sim_path     = np_root / "ch11-metrics"     / "similarity_matrices.npz"
    ch12_path    = np_root / "ch12-label-aware" / "ch12_summary.csv"
    ch13_path    = np_root / "ch13-label-blind" / "ch13_summary.csv"
    output_path  = np_root / "ch14-comparison"
    output_path.mkdir(parents = True, exist_ok = True)

    assert sim_path.exists(),  f"Run ch11 first; missing {sim_path}"
    assert ch12_path.exists(), f"Run ch12 first; missing {ch12_path}"
    assert ch13_path.exists(), f"Run ch13 first; missing {ch13_path}"

    data         = np.load(sim_path, allow_pickle = False)
    labels       = data["labels"]
    metric_names = ["pearson_raw", "lagged_xcorr_max", "envelope_corr",
                    "log_spectrum_corr", "mfcc_cosine"]

    ch12_df      = pd.read_csv(ch12_path).set_index("metric")
    ch13_df      = pd.read_csv(ch13_path)

    # -----------------------------------------------------------------------
    # Step 1: Per-metric axis scores.
    # -----------------------------------------------------------------------
    rows = []
    for metric in metric_names:
        sim       = data[metric]
        tt, off   = split_tt_off(sim, labels)

        # Target cohesion z-score: how many SDs above the off-diagonal mean does
        # the within-target mean sit?  Standardizes across very different metric
        # scales (Pearson r ~ 0 vs envelope r ~ 0.74).
        off_mean  = float(off.mean())
        off_sd    = float(off.std(ddof = 1))
        cohesion_z = (float(tt.mean()) - off_mean) / off_sd if off_sd > 0 else 0.0

        # Label separation: Cohen's d (tt vs dd).
        d_tt_dd   = float(ch12_df.loc[metric, "cohens_d_tt_vs_dd"])

        # Intrinsic cluster strength: best silhouette over the two methods.
        sil_rows  = ch13_df[ch13_df["metric"] == metric]
        sil_best  = float(sil_rows["silhouette"].max())
        sil_method= sil_rows.loc[sil_rows["silhouette"].idxmax(), "method"]
        ari_best  = float(sil_rows["ari_vs_labels"].max())

        rows.append({
            "metric":                       metric,
            "target_cohesion_z":            cohesion_z,
            "label_separation_cohens_d":    d_tt_dd,
            "intrinsic_silhouette_best":    sil_best,
            "intrinsic_silhouette_method":  sil_method,
            "max_ari_vs_labels":            ari_best,
        })

    summary_df = pd.DataFrame(rows)

    # Per-axis rank (1 = best).
    summary_df["rank_target_cohesion"]  = summary_df["target_cohesion_z"].rank(ascending = False).astype(int)
    summary_df["rank_label_separation"] = summary_df["label_separation_cohens_d"].rank(ascending = False).astype(int)
    summary_df["rank_intrinsic_clust"]  = summary_df["intrinsic_silhouette_best"].rank(ascending = False).astype(int)

    summary_df.to_csv(output_path / "ch14_summary_ranked.csv", index = False)
    print("Per-axis scores:", flush = True)
    print(summary_df.to_string(index = False), flush = True)

    # -----------------------------------------------------------------------
    # Step 2: Cross-metric Mantel-style correlation.
    # -----------------------------------------------------------------------
    flats = {m: upper_tri_flat(data[m]).astype(np.float64) for m in metric_names}
    n_m   = len(metric_names)
    mantel = np.eye(n_m, dtype = np.float64)
    for i, mi in enumerate(metric_names):
        for j, mj in enumerate(metric_names):
            if j <= i:
                continue
            r          = float(np.corrcoef(flats[mi], flats[mj])[0, 1])
            mantel[i, j] = r
            mantel[j, i] = r

    mantel_df = pd.DataFrame(mantel, index = metric_names, columns = metric_names)
    mantel_df.to_csv(output_path / "ch14_cross_metric_mantel.csv")
    print("Cross-metric Mantel-style Pearson r between similarity vectors:", flush = True)
    print(mantel_df.round(4).to_string(), flush = True)

    # -----------------------------------------------------------------------
    # Step 3: Plots.
    # -----------------------------------------------------------------------
    # Mantel heatmap.
    fig, ax = plt.subplots(figsize = (6.5, 5.5))
    im = ax.imshow(mantel, cmap = "RdBu_r", vmin = -1, vmax = 1)
    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_m))
    ax.set_xticklabels(metric_names, rotation = 30, ha = "right", fontsize = 9)
    ax.set_yticklabels(metric_names, fontsize = 9)
    for i in range(n_m):
        for j in range(n_m):
            ax.text(j, i, f"{mantel[i, j]:+.2f}", ha = "center", va = "center",
                    fontsize = 8,
                    color = "white" if abs(mantel[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax = ax, label = "Pearson r over 44850 pairs")
    ax.set_title("Cross-metric similarity-vector correlation")
    fig.tight_layout()
    fig.savefig(output_path / "ch14_cross_metric_mantel.png", dpi = 150)
    plt.close(fig)

    # Axis-ranking bar chart (3 panels).
    fig, axes = plt.subplots(1, 3, figsize = (15, 4.5))
    x = np.arange(n_m)

    ax = axes[0]
    vals = summary_df.set_index("metric").loc[metric_names, "target_cohesion_z"].to_numpy()
    ax.bar(x, vals, color = "tomato", edgecolor = "black")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("z-score (mean_tt vs off-diagonal)")
    ax.set_title("Target cohesion (higher = targets more mutually similar)")

    ax = axes[1]
    vals = summary_df.set_index("metric").loc[metric_names, "label_separation_cohens_d"].to_numpy()
    ax.bar(x, vals, color = "darkorange", edgecolor = "black")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("Cohen's d (tt vs dd)")
    ax.set_title("Label separation (higher = tt > dd)")

    ax = axes[2]
    vals = summary_df.set_index("metric").loc[metric_names, "intrinsic_silhouette_best"].to_numpy()
    ax.bar(x, vals, color = "steelblue", edgecolor = "black")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("Best silhouette (k=2)")
    ax.set_title("Intrinsic cluster strength (label-blind)")

    fig.tight_layout()
    fig.savefig(output_path / "ch14_axis_rankings.png", dpi = 150)
    plt.close(fig)

    print(f"Chapter 14 done.  Outputs saved to: {output_path}", flush = True)


if __name__ == "__main__":
    main()
