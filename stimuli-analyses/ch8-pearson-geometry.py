"""Chapter 8. Pearson-Based Geometry of the Full Stimulus Set.

Goal
----
Map the shape of the 300-stimulus set using pairwise Pearson similarities.
First label-blind: where do stimuli sit relative to one another, what natural
clusters emerge, how strong are they?  Then compare those emergent clusters
to the experimenter target/distractor labels.

Pipeline
--------
1. Build the 300 x 300 Pearson similarity matrix on raw centered waveforms.
2. Convert to a non-negative distance matrix d = 1 - r (then clipped).
3. Embed the distance matrix in 2-D via:
     - classical MDS  (linear; preserves global geometry under r-derived
       distances, computed via eigendecomposition of the double-centered
       squared-distance matrix)
     - PCA            (linear; on the centered waveform matrix directly,
       which is the same geometry up to a scale per row)
     - t-SNE          (non-linear; for cluster visualization only)
4. Run clustering at k = 2, 3, 4, 5 with two methods (agglomerative average
   linkage on distance, spectral on similarity).  Score each with silhouette
   (intrinsic) and adjusted Rand index against experimenter labels (extrinsic
   reality check).
5. Test whether the target/distractor split is itself a dominant axis: project
   each stimulus onto the difference-of-class-means direction (a 1-D LDA-like
   axis) and report the variance explained / Fisher discriminant ratio.

Outputs
-------
outputs/pearson/ch8-geometry/
    ch8_distance_matrix.npz          - distance + similarity + labels + nums
    ch8_clustering_summary.csv       - per (method, k): silhouette, ARI, sizes
    ch8_label_axis_summary.csv       - Fisher ratio + var-explained for the
                                        target-mean - distractor-mean direction
    ch8_assignments.csv              - per stimulus: cluster ids for each (method, k)
    ch8_mds.png                      - 2-D MDS, two panels: by labels / by k=2 cluster
    ch8_pca.png                      - 2-D PCA, two panels: by labels / by k=2 cluster
    ch8_tsne.png                     - 2-D t-SNE, two panels
    ch8_silhouette_ari.png           - bar chart across k for each method
    ch8_label_axis_projection.png    - histogram of LDA-axis projections by label
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io       import wavfile
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def load_stimuli(cur_path):
    """Load all target and distractor waveforms (raw on-disk PCM, centered).

    Args:
        cur_path: Path to the script directory.

    Returns:
        Tuple (waveform_matrix, labels, numbers, sr).
    """
    stimuli_path  = cur_path / ".." / "raw_data" / "audio_stimuli"
    fs_targs_path = stimuli_path / "full_sentence"     / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"
    fs_dists_path = stimuli_path / "full_sentence"     / "distractors"
    is_dists_path = stimuli_path / "imagined_sentence" / "distractors"

    rows    = []
    labels  = []
    numbers = []
    sr_seen = None
    for label, directory in [("target", fs_targs_path),
                             ("target", is_targs_path),
                             ("distractor", fs_dists_path),
                             ("distractor", is_dists_path)]:
        for wav_path in sorted(directory.glob("*.wav")):
            sr, raw = wavfile.read(wav_path)
            if sr_seen is None:
                sr_seen = sr
            assert sr == sr_seen, f"Sample rate mismatch in {wav_path}"
            rows.append(raw.astype(np.float64))
            labels.append(label)
            numbers.append(int(wav_path.stem))

    waveform_matrix = np.vstack(rows)
    waveform_matrix = waveform_matrix - waveform_matrix.mean(axis = 1, keepdims = True)
    print(f"Loaded {waveform_matrix.shape[0]} stimuli, length L={waveform_matrix.shape[1]}, "
          f"sr={sr_seen} Hz (raw on-disk, centered).", flush = True)
    return waveform_matrix, np.array(labels), np.array(numbers, dtype = np.int64), sr_seen


def classical_mds(distance_matrix, n_components = 2):
    """Classical (Torgerson) MDS via eigendecomposition of the Gram matrix.

    Args:
        distance_matrix: (n, n) symmetric non-negative distance matrix.
        n_components:    Embedding dimensionality.

    Returns:
        (n, n_components) embedding, plus 1-D array of all eigenvalues
        (descending) for variance-explained reporting.
    """
    n         = distance_matrix.shape[0]
    d_sq      = distance_matrix ** 2
    j_center  = np.eye(n) - np.ones((n, n)) / n
    gram      = -0.5 * j_center @ d_sq @ j_center
    gram      = 0.5 * (gram + gram.T)                   # symmetrize numerical noise

    eigvals, eigvecs = np.linalg.eigh(gram)
    order            = np.argsort(eigvals)[::-1]
    eigvals          = eigvals[order]
    eigvecs          = eigvecs[:, order]

    pos      = np.maximum(eigvals[:n_components], 0)
    embed    = eigvecs[:, :n_components] * np.sqrt(pos)[np.newaxis, :]
    return embed, eigvals


def fisher_discriminant_ratio(scores, labels):
    """Fisher discriminant ratio for a 1-D projection given two labels.

    Ratio = (mean_a - mean_b)^2 / (var_a + var_b).

    Args:
        scores: (n,) projection values.
        labels: (n,) array with two unique values.

    Returns:
        Float Fisher ratio.
    """
    uniq = np.unique(labels)
    assert len(uniq) == 2, "Fisher ratio requires exactly two labels"
    a = scores[labels == uniq[0]]
    b = scores[labels == uniq[1]]
    num = (a.mean() - b.mean()) ** 2
    den = a.var(ddof = 1) + b.var(ddof = 1)
    if den == 0:
        return float("inf")
    return float(num / den)


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path    = Path(__file__).resolve().parent
    output_path = cur_path / "outputs" / "pearson" / "ch8-geometry"
    output_path.mkdir(parents = True, exist_ok = True)

    waveform_matrix, labels, numbers, _ = load_stimuli(cur_path)
    n        = waveform_matrix.shape[0]
    label_int = (labels == "target").astype(int)

    # -----------------------------------------------------------------------
    # Step 1: Pearson similarity + distance.
    # -----------------------------------------------------------------------
    sim       = np.corrcoef(waveform_matrix).astype(np.float64)
    sim       = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)

    dist      = np.clip(1.0 - sim, 0.0, None)
    dist      = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    np.savez(output_path / "ch8_distance_matrix.npz",
             similarity        = sim.astype(np.float32),
             distance          = dist.astype(np.float32),
             labels            = labels,
             stimulus_numbers  = numbers)

    print(f"Pearson similarity: mean off-diag = {sim[np.triu_indices(n, k = 1)].mean():+.5f}  "
          f"std = {sim[np.triu_indices(n, k = 1)].std():.5f}", flush = True)

    # -----------------------------------------------------------------------
    # Step 2: Embeddings.
    # -----------------------------------------------------------------------
    print("Computing classical MDS ...", flush = True)
    mds_embed, mds_eigs = classical_mds(dist, n_components = 2)
    pos_eigs = np.maximum(mds_eigs, 0)
    var_explained_mds = pos_eigs[:2].sum() / pos_eigs.sum() if pos_eigs.sum() > 0 else 0.0
    print(f"  MDS first 2 components explain {var_explained_mds * 100:.2f}% of "
          f"positive Gram variance.", flush = True)

    print("Computing PCA on centered waveforms ...", flush = True)
    pca = PCA(n_components = 2)
    pca_embed = pca.fit_transform(waveform_matrix)
    print(f"  PCA first 2 components explain "
          f"{pca.explained_variance_ratio_.sum() * 100:.2f}% of waveform variance.",
          flush = True)

    print("Computing t-SNE on Pearson distance ...", flush = True)
    tsne = TSNE(n_components = 2, metric = "precomputed", init = "random",
                perplexity = 30, random_state = 0, max_iter = 1000)
    tsne_embed = tsne.fit_transform(dist)

    # -----------------------------------------------------------------------
    # Step 3: Clustering at k = 2, 3, 4, 5.
    # -----------------------------------------------------------------------
    cluster_summary = []
    assignments     = {"stimulus_number": numbers, "label": labels}

    # Spectral clustering needs a non-negative affinity.
    aff = sim.copy()
    lo  = aff.min()
    if lo < 0:
        aff = (aff - lo) / (1.0 - lo)
    aff = np.clip(aff, 0.0, 1.0)
    aff = 0.5 * (aff + aff.T)
    np.fill_diagonal(aff, 1.0)

    for k in [2, 3, 4, 5]:
        agglo = AgglomerativeClustering(n_clusters = k, metric = "precomputed",
                                        linkage = "average")
        agglo_labels = agglo.fit_predict(dist)

        try:
            spec = SpectralClustering(n_clusters = k, affinity = "precomputed",
                                      random_state = 0, assign_labels = "kmeans")
            spec_labels = spec.fit_predict(aff)
        except Exception as exc:
            print(f"  spectral k={k} failed: {exc}", flush = True)
            spec_labels = np.zeros(n, dtype = int)

        def safe_silhouette(cl):
            uniq = np.unique(cl)
            if len(uniq) < 2 or any(np.sum(cl == u) < 2 for u in uniq):
                return float("nan")
            return float(silhouette_score(dist, cl, metric = "precomputed"))

        for method, cl in [("agglomerative_avg", agglo_labels),
                           ("spectral_kmeans",   spec_labels)]:
            sizes = np.bincount(cl, minlength = k).tolist()
            sil   = safe_silhouette(cl)
            ari   = float(adjusted_rand_score(label_int, cl)) if k == 2 else float(
                        adjusted_rand_score(label_int, cl))
            cluster_summary.append({
                "method":     method,
                "k":          k,
                "sizes":      sizes,
                "silhouette": sil,
                "ari_vs_labels": ari,
            })
            assignments[f"{method}_k{k}"] = cl
            print(f"  {method:18s} k={k}  sizes={sizes}  silhouette={sil:.4f}  "
                  f"ARI={ari:+.4f}", flush = True)

    pd.DataFrame(cluster_summary).to_csv(output_path / "ch8_clustering_summary.csv",
                                         index = False)
    pd.DataFrame(assignments).to_csv(output_path / "ch8_assignments.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 4: Is the target/distractor split a dominant axis?
    # -----------------------------------------------------------------------
    is_t = labels == "target"
    is_d = ~is_t
    mean_t = waveform_matrix[is_t].mean(axis = 0)
    mean_d = waveform_matrix[is_d].mean(axis = 0)
    direction = mean_t - mean_d
    dir_norm  = np.linalg.norm(direction)
    assert dir_norm > 0, "Target and distractor mean waveforms are identical"
    direction = direction / dir_norm

    # Project each stimulus onto the unit difference-of-means direction.
    proj      = waveform_matrix @ direction                     # (n,)
    fisher    = fisher_discriminant_ratio(proj, labels)

    # Variance explained by this direction relative to total waveform variance.
    total_var      = float(np.sum(waveform_matrix.var(axis = 0)))
    proj_var       = float(proj.var(ddof = 1))
    frac_explained = proj_var / total_var if total_var > 0 else 0.0

    # Compare to a null: project onto random unit directions.
    rng       = np.random.default_rng(0)
    null_fisher = np.empty(2000, dtype = np.float64)
    null_var    = np.empty(2000, dtype = np.float64)
    L         = waveform_matrix.shape[1]
    for i in range(2000):
        v          = rng.standard_normal(L)
        v         /= np.linalg.norm(v)
        rp         = waveform_matrix @ v
        null_fisher[i] = fisher_discriminant_ratio(rp, labels)
        null_var[i]    = rp.var(ddof = 1) / total_var
    p_fisher = (np.sum(null_fisher >= fisher) + 1) / (len(null_fisher) + 1)
    p_var    = (np.sum(null_var    >= frac_explained) + 1) / (len(null_var) + 1)

    print(f"Label axis Fisher discriminant ratio = {fisher:.4f}  "
          f"(null mean={null_fisher.mean():.4f}  p={p_fisher:.4g})", flush = True)
    print(f"Label axis variance fraction        = {frac_explained * 100:.4f}%  "
          f"(null mean={null_var.mean() * 100:.4f}%  p={p_var:.4g})", flush = True)

    pd.DataFrame([{
        "fisher_ratio_label_axis":    fisher,
        "fisher_null_mean":           float(null_fisher.mean()),
        "fisher_null_std":            float(null_fisher.std(ddof = 1)),
        "fisher_p_random_direction":  float(p_fisher),
        "var_fraction_label_axis":    frac_explained,
        "var_null_mean":              float(null_var.mean()),
        "var_null_std":               float(null_var.std(ddof = 1)),
        "var_p_random_direction":     float(p_var),
    }]).to_csv(output_path / "ch8_label_axis_summary.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 5: Plots.
    # -----------------------------------------------------------------------
    def two_panel(embed, name, k2_clusters):
        fig, axes = plt.subplots(1, 2, figsize = (12, 5))
        ax = axes[0]
        for lab, color in [("target", "tomato"), ("distractor", "steelblue")]:
            mask = labels == lab
            ax.scatter(embed[mask, 0], embed[mask, 1], s = 18, alpha = 0.7,
                       color = color, label = lab,
                       edgecolor = "black", linewidth = 0.3)
        ax.set_title(f"{name}: experimenter labels")
        ax.legend()

        ax = axes[1]
        for cl, color in [(0, "darkorange"), (1, "purple")]:
            mask = k2_clusters == cl
            ax.scatter(embed[mask, 0], embed[mask, 1], s = 18, alpha = 0.7,
                       color = color, label = f"cluster {cl} (n={mask.sum()})",
                       edgecolor = "black", linewidth = 0.3)
        ax.set_title(f"{name}: agglomerative k=2 clusters")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path / f"ch8_{name.lower()}.png", dpi = 150)
        plt.close(fig)

    agglo_k2 = np.array(assignments["agglomerative_avg_k2"])
    two_panel(mds_embed,  "MDS",  agglo_k2)
    two_panel(pca_embed,  "PCA",  agglo_k2)
    two_panel(tsne_embed, "tSNE", agglo_k2)

    # Silhouette + ARI bar chart (across k for each method).
    summary_df = pd.DataFrame(cluster_summary)
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    for method, color in [("agglomerative_avg", "tomato"),
                          ("spectral_kmeans",   "steelblue")]:
        sub = summary_df[summary_df["method"] == method]
        axes[0].plot(sub["k"], sub["silhouette"], marker = "o", color = color,
                     linewidth = 2, label = method)
        axes[1].plot(sub["k"], sub["ari_vs_labels"], marker = "o", color = color,
                     linewidth = 2, label = method)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette score (intrinsic)")
    axes[0].set_title("Cluster cohesion vs k")
    axes[0].legend()
    axes[1].axhline(0, color = "black", linewidth = 0.6)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Adjusted Rand Index vs labels")
    axes[1].set_title("Agreement with experimenter labels vs k")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path / "ch8_silhouette_ari.png", dpi = 150)
    plt.close(fig)

    # Label-axis projection histogram.
    fig, ax = plt.subplots(figsize = (8, 4.5))
    ax.hist(proj[is_t], bins = 30, color = "tomato",    alpha = 0.6,
            edgecolor = "black", label = f"target (n={is_t.sum()})")
    ax.hist(proj[is_d], bins = 30, color = "steelblue", alpha = 0.6,
            edgecolor = "black", label = f"distractor (n={is_d.sum()})")
    ax.axvline(proj[is_t].mean(), color = "tomato",    linewidth = 2)
    ax.axvline(proj[is_d].mean(), color = "steelblue", linewidth = 2)
    ax.set_xlabel("Projection onto (mean_target - mean_distractor) direction")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Label-discriminant axis  Fisher ratio={fisher:.3f}  "
                 f"var fraction={frac_explained * 100:.3f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "ch8_label_axis_projection.png", dpi = 150)
    plt.close(fig)

    print(f"Chapter 8 done.  Outputs saved to: {output_path}", flush = True)


if __name__ == "__main__":
    main()
