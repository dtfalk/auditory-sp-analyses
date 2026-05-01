"""Chapter 13. Label-Blind Structure Under Alternative Metrics.

Goal
----
Treat each similarity matrix from ch11 as a stand-alone affinity over 300
clips, ignore the target/distractor labels, and ask whether natural clusters
emerge.  Compare the discovered clusters back to the experimenter labels.

Approach
--------
For every metric:
  1. Convert similarity to distance (d = 1 - similarity for [-1,1] / [0,1]
     metrics; we clip to >= 0).
  2. Run two clustering procedures with k = 2:
       a. Agglomerative clustering on the distance matrix (average linkage).
       b. Spectral clustering on the similarity matrix (works directly with
          a symmetric non-negative affinity).
  3. Score each clustering with:
       - silhouette score (on the distance matrix; intrinsic; higher = better)
       - Adjusted Rand Index (ARI) vs the target/distractor labels (extrinsic;
         0 = chance, 1 = perfect agreement, negative = worse than chance).
  4. Embed the similarity in 2-D via t-SNE on the distance matrix and color-
     code by (a) experimenter labels and (b) discovered clusters.

Outputs
-------
outputs/non_pearson/ch13-label-blind/
    ch13_summary.csv                  - one row per metric x clustering method
    ch13_assignments_<metric>.csv     - per-stimulus cluster assignments
    ch13_embedding_<metric>.png       - 2x1 t-SNE colored by labels / clusters
    ch13_silhouette_ari_summary.png   - bar chart across metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def similarity_to_distance(sim_matrix, metric_name):
    """Convert a similarity matrix into a non-negative symmetric distance matrix.

    Args:
        sim_matrix:  (n, n) similarity matrix.
        metric_name: Name of the metric (for diagnostic logging).

    Returns:
        (n, n) float64 distance matrix with zero diagonal.
    """
    dist = 1.0 - sim_matrix.astype(np.float64)
    dist = np.clip(dist, 0.0, None)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def affinity_from_similarity(sim_matrix):
    """Convert similarity into a non-negative affinity for spectral clustering.

    Shifts and clips so the affinity is in [0, 1] with self-affinity 1.

    Args:
        sim_matrix: (n, n) similarity matrix.

    Returns:
        (n, n) float64 non-negative symmetric affinity matrix.
    """
    aff = sim_matrix.astype(np.float64)
    lo  = aff.min()
    if lo < 0:
        aff = (aff - lo) / (1.0 - lo)         # rescale into [0, 1] preserving order
    aff = np.clip(aff, 0.0, 1.0)
    aff = 0.5 * (aff + aff.T)
    np.fill_diagonal(aff, 1.0)
    return aff


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path    = Path(__file__).resolve().parent
    in_path     = cur_path / "outputs" / "non_pearson" / "ch11-metrics" / "similarity_matrices.npz"
    output_path = cur_path / "outputs" / "non_pearson" / "ch13-label-blind"
    output_path.mkdir(parents = True, exist_ok = True)

    assert in_path.exists(), f"Run ch11 first; missing {in_path}"
    data            = np.load(in_path, allow_pickle = False)
    labels          = data["labels"]
    stimulus_nums   = data["stimulus_numbers"]
    metric_names    = ["pearson_raw", "lagged_xcorr_max", "envelope_corr",
                       "log_spectrum_corr", "mfcc_cosine"]

    label_int       = (labels == "target").astype(int)        # 0/1 vector for ARI

    summary_rows    = []

    for metric in metric_names:
        sim     = data[metric]
        dist    = similarity_to_distance(sim, metric)
        aff     = affinity_from_similarity(sim)

        print(f"--- {metric} ---", flush = True)

        # 1. Agglomerative clustering on the distance matrix (avg linkage).
        agglo = AgglomerativeClustering(n_clusters = 2, metric = "precomputed",
                                        linkage = "average")
        agglo_labels = agglo.fit_predict(dist)

        # 2. Spectral clustering on the affinity matrix.
        try:
            spec = SpectralClustering(n_clusters = 2, affinity = "precomputed",
                                      random_state = 0, assign_labels = "kmeans")
            spec_labels = spec.fit_predict(aff)
        except Exception as exc:                                # rare numerical failure
            print(f"    spectral failed: {exc}", flush = True)
            spec_labels = np.zeros_like(agglo_labels)

        # Silhouette: requires at least 2 distinct clusters with > 1 member each.
        def safe_silhouette(cluster_labels):
            uniq = np.unique(cluster_labels)
            if len(uniq) < 2 or any(np.sum(cluster_labels == u) < 2 for u in uniq):
                return float("nan")
            return float(silhouette_score(dist, cluster_labels, metric = "precomputed"))

        sil_agglo = safe_silhouette(agglo_labels)
        sil_spec  = safe_silhouette(spec_labels)
        ari_agglo = float(adjusted_rand_score(label_int, agglo_labels))
        ari_spec  = float(adjusted_rand_score(label_int, spec_labels))

        # Cluster sizes.
        agg_sizes  = np.bincount(agglo_labels, minlength = 2)
        spec_sizes = np.bincount(spec_labels,  minlength = 2)

        print(f"  agglomerative  sizes={agg_sizes.tolist()}  "
              f"silhouette={sil_agglo:.4f}  ARI_vs_labels={ari_agglo:+.4f}", flush = True)
        print(f"  spectral       sizes={spec_sizes.tolist()}  "
              f"silhouette={sil_spec:.4f}  ARI_vs_labels={ari_spec:+.4f}", flush = True)

        summary_rows.append({"metric": metric, "method": "agglomerative_avg",
                             "size_0": int(agg_sizes[0]), "size_1": int(agg_sizes[1]),
                             "silhouette": sil_agglo, "ari_vs_labels": ari_agglo})
        summary_rows.append({"metric": metric, "method": "spectral_kmeans",
                             "size_0": int(spec_sizes[0]), "size_1": int(spec_sizes[1]),
                             "silhouette": sil_spec, "ari_vs_labels": ari_spec})

        # Per-stimulus assignments.
        pd.DataFrame({
            "stimulus_number":   stimulus_nums,
            "label":             labels,
            "agglomerative_avg": agglo_labels,
            "spectral_kmeans":   spec_labels,
        }).to_csv(output_path / f"ch13_assignments_{metric}.csv", index = False)

        # 3. t-SNE embedding for visualization.
        try:
            tsne = TSNE(n_components = 2, metric = "precomputed", init = "random",
                        perplexity = 30, random_state = 0, max_iter = 1000)
            embed = tsne.fit_transform(dist)
        except Exception as exc:
            print(f"    t-SNE failed: {exc}", flush = True)
            embed = np.zeros((dist.shape[0], 2))

        fig, axes = plt.subplots(1, 2, figsize = (12, 5))

        ax = axes[0]
        for lab, color in [("target", "tomato"), ("distractor", "steelblue")]:
            mask = labels == lab
            ax.scatter(embed[mask, 0], embed[mask, 1], s = 16, alpha = 0.7,
                       color = color, label = lab, edgecolor = "black", linewidth = 0.3)
        ax.set_title(f"{metric}: t-SNE colored by experimenter labels")
        ax.legend()

        ax = axes[1]
        for cl, color in [(0, "darkorange"), (1, "purple")]:
            mask = agglo_labels == cl
            ax.scatter(embed[mask, 0], embed[mask, 1], s = 16, alpha = 0.7,
                       color = color,
                       label = f"agglo cluster {cl} (n={mask.sum()})",
                       edgecolor = "black", linewidth = 0.3)
        ax.set_title(f"{metric}: t-SNE colored by agglomerative clusters "
                     f"(ARI={ari_agglo:+.3f})")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path / f"ch13_embedding_{metric}.png", dpi = 150)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / "ch13_summary.csv", index = False)

    # -----------------------------------------------------------------------
    # Cross-metric summary bar chart.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    x      = np.arange(len(metric_names))
    width  = 0.4
    sil_a  = summary_df[summary_df["method"] == "agglomerative_avg"].set_index("metric").loc[metric_names, "silhouette"].to_numpy()
    sil_s  = summary_df[summary_df["method"] == "spectral_kmeans"].set_index("metric").loc[metric_names, "silhouette"].to_numpy()
    ari_a  = summary_df[summary_df["method"] == "agglomerative_avg"].set_index("metric").loc[metric_names, "ari_vs_labels"].to_numpy()
    ari_s  = summary_df[summary_df["method"] == "spectral_kmeans"].set_index("metric").loc[metric_names, "ari_vs_labels"].to_numpy()

    ax = axes[0]
    ax.bar(x - width / 2, sil_a, width, color = "tomato",    label = "agglomerative")
    ax.bar(x + width / 2, sil_s, width, color = "steelblue", label = "spectral")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Intrinsic cluster validity (k=2)")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, ari_a, width, color = "tomato",    label = "agglomerative")
    ax.bar(x + width / 2, ari_s, width, color = "steelblue", label = "spectral")
    ax.axhline(0, color = "black", linewidth = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation = 20, ha = "right", fontsize = 9)
    ax.set_ylabel("Adjusted Rand Index vs labels")
    ax.set_title("Agreement of discovered clusters with experimenter labels")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path / "ch13_silhouette_ari_summary.png", dpi = 150)
    plt.close(fig)

    print(f"Chapter 13 done.  Outputs saved to: {output_path}", flush = True)


if __name__ == "__main__":
    main()
