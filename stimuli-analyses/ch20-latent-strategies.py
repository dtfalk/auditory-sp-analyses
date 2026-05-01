"""Chapter 20.  Latent response strategies across subjects.

Question
--------
Do subjects cluster into qualitatively different *strategies* -- e.g. a group
that tracks the experimenter labels and a group that calls everything a target?

Two complementary views:

  (A) Cluster subjects by their full response vector over the 150 stimuli of
      each block (Spectral clustering, k = 2..4) and report per-cluster SDT
      summaries.

  (B) Cluster subjects by their fitted logistic-regression coefficient profile
      from chapter 18 (one weight per metric contrast).  This asks whether
      different subjects rely on different acoustic cues, even when their
      overall accuracy is similar.

Outputs
-------
outputs/book2/ch20-strategies/
    ch20_response_clusters.csv
    ch20_coef_clusters.csv
    ch20_subject_response_pca.png
    ch20_subject_coef_heatmap.png
    ch20_cluster_sdt.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.cluster        import SpectralClustering, KMeans
from sklearn.decomposition  import PCA
from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import (load_all_trials, load_similarity_matrices,
                           build_subject_stim_matrix,
                           build_stimulus_block_map, block_feature_table,
                           METRICS)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book2" / "ch20-strategies"
OUT_PATH.mkdir(parents = True, exist_ok = True)

RNG = np.random.default_rng(0)


# ============================================================================
# Helpers
# ============================================================================

def per_subject_sdt_simple(y, lbl):
    n_t = (lbl == 1).sum()
    n_d = (lbl == 0).sum()
    hr = ((y == 1) & (lbl == 1)).sum() / max(n_t, 1)
    fr = ((y == 1) & (lbl == 0)).sum() / max(n_d, 1)
    acc = (y == lbl).mean()
    return hr, fr, acc


# ============================================================================
# Load
# ============================================================================

print("Loading data ...", flush = True)
trials    = load_all_trials()
sim_dict  = load_similarity_matrices()
block_map = build_stimulus_block_map()
feat_per_block = {b: block_feature_table(trials, block_map, sim_dict, b)
                  for b in ("full_sentence", "imagined_sentence")}


# ============================================================================
# (A) Cluster subjects by response vectors, per block
# ============================================================================

resp_rows = []
fig_pca, axes_pca = plt.subplots(1, 2, figsize = (12, 5.5))
for ax, block in zip(axes_pca, ("full_sentence", "imagined_sentence")):
    Y, subjects, stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    # mean-impute any NaN cells (very rare)
    Yi = np.where(np.isnan(Y), np.nanmean(Y, axis = 0), Y)

    # Spectral clustering on cosine similarity between subject response vectors
    Yc = Yi - Yi.mean(axis = 1, keepdims = True)
    norm = np.linalg.norm(Yc, axis = 1, keepdims = True) + 1e-9
    cos = (Yc @ Yc.T) / (norm @ norm.T)
    aff = (cos + 1.0) / 2.0  # to [0, 1]
    np.fill_diagonal(aff, 1.0)

    for k in (2, 3, 4):
        sc = SpectralClustering(n_clusters = k, affinity = "precomputed",
                                random_state = 0, assign_labels = "kmeans")
        labels_sc = sc.fit_predict(aff)
        for s_idx, subj in enumerate(subjects):
            hr, fr, acc = per_subject_sdt_simple(Yi[s_idx], true_labels)
            resp_rows.append(dict(block_type = block, subject = subj, k = k,
                                  cluster = int(labels_sc[s_idx]),
                                  hit_rate = hr, fa_rate = fr, accuracy = acc,
                                  response_rate = float(np.mean(Yi[s_idx]))))

    # PCA visualisation (k=2 colouring)
    pcs = PCA(n_components = 2, random_state = 0).fit_transform(Yi)
    sc2 = SpectralClustering(n_clusters = 2, affinity = "precomputed",
                             random_state = 0, assign_labels = "kmeans")
    lab2 = sc2.fit_predict(aff)
    for c in (0, 1):
        sel = lab2 == c
        ax.scatter(pcs[sel, 0], pcs[sel, 1], s = 90, edgecolor = "k",
                   label = f"cluster {c}  (n={sel.sum()})", alpha = 0.85)
    for i, subj in enumerate(subjects):
        ax.annotate(subj[-3:], (pcs[i, 0], pcs[i, 1]), fontsize = 6,
                    xytext = (3, 2), textcoords = "offset points")
    ax.set_title(f"{block}: subjects in PCA(response vector) space, k=2")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize = 8)
plt.suptitle("Latent response-strategy clusters", fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch20_subject_response_pca.png", dpi = 130)
plt.close()

resp_df = pd.DataFrame(resp_rows)
resp_df.to_csv(OUT_PATH / "ch20_response_clusters.csv", index = False)


# ============================================================================
# (B) Cluster subjects by logistic-regression coefficient profile
# ============================================================================

print("Fitting per-subject logistic to recover metric weights ...", flush = True)
coef_rows = []
for blk in ("full_sentence", "imagined_sentence"):
    feat = feat_per_block[blk]
    feat_lookup = feat.set_index("stimulus_number")
    sub_trials = trials[trials["block_type"] == blk]
    for subj, g in sub_trials.groupby("subject"):
        X = feat_lookup.loc[g["stimulus_number"].to_numpy(),
                            [f"{m}_contrast" for m in METRICS]].to_numpy(float)
        y = g["called_target"].to_numpy(int)
        if y.min() == y.max():
            continue
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter = 1000, C = 1.0).fit(Xs, y)
        row = dict(block_type = blk, subject = subj,
                   intercept = float(clf.intercept_[0]))
        for m, c in zip(METRICS, clf.coef_[0]):
            row[f"w_{m}"] = float(c)
        coef_rows.append(row)
coef_df = pd.DataFrame(coef_rows)

# cluster on weight profiles via KMeans, k=2,3,4
W = coef_df[[f"w_{m}" for m in METRICS]].to_numpy()
Wz = StandardScaler().fit_transform(W)
for k in (2, 3, 4):
    km = KMeans(n_clusters = k, n_init = 20, random_state = 0).fit(Wz)
    coef_df[f"cluster_k{k}"] = km.labels_
coef_df.to_csv(OUT_PATH / "ch20_coef_clusters.csv", index = False)

# heatmap of coefficients sorted by k=2 cluster
fig, axes = plt.subplots(1, 2, figsize = (14, max(6, 0.3 * len(coef_df))))
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = coef_df[coef_df["block_type"] == blk].copy()
    sub = sub.sort_values(["cluster_k2", "subject"])
    M = sub[[f"w_{m}" for m in METRICS]].to_numpy()
    im = ax.imshow(M, cmap = "RdBu_r", vmin = -2, vmax = 2, aspect = "auto")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["subject"].to_numpy(), fontsize = 6)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation = 30, ha = "right", fontsize = 8)
    # cluster boundaries
    boundaries = np.where(np.diff(sub["cluster_k2"].to_numpy()) != 0)[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color = "k", lw = 0.8)
    ax.set_title(f"{blk}: per-subject metric weights (sorted by k=2 cluster)")
    plt.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04, label = "weight")
plt.tight_layout()
plt.savefig(OUT_PATH / "ch20_subject_coef_heatmap.png", dpi = 130)
plt.close()


# ============================================================================
# Cluster-level SDT summary (k=2 from the response-vector clustering)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize = (12, 5.5), sharey = True)
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = resp_df[(resp_df["block_type"] == blk) & (resp_df["k"] == 2)]
    for c, g in sub.groupby("cluster"):
        ax.scatter(g["fa_rate"], g["hit_rate"], s = 80, edgecolor = "k",
                   alpha = 0.85, label = f"cluster {c} (n={len(g)}, "
                                          f"resp_rate={g['response_rate'].mean():.2f})")
    ax.plot([0, 1], [0, 1], "--", color = "gray", lw = 0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FA rate"); ax.set_ylabel("Hit rate")
    ax.set_title(f"{blk}: cluster-level ROC points (k=2)")
    ax.legend(fontsize = 8)
plt.suptitle("Per-cluster SDT view of latent strategy groups", fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch20_cluster_sdt.png", dpi = 130)
plt.close()

# print quick textual summary
for blk in ("full_sentence", "imagined_sentence"):
    print(f"\n[{blk}]  k=2 cluster summary:", flush = True)
    sub = resp_df[(resp_df["block_type"] == blk) & (resp_df["k"] == 2)]
    g = sub.groupby("cluster").agg(n = ("subject", "size"),
                                   resp_rate = ("response_rate", "mean"),
                                   hit = ("hit_rate", "mean"),
                                   fa  = ("fa_rate", "mean"),
                                   acc = ("accuracy", "mean"))
    print(g.to_string(), flush = True)

print(f"\nChapter 20 done.  Outputs saved to: {OUT_PATH}", flush = True)
