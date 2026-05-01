"""Chapter 19.  Response-derived stimulus geometry and RSA.

Question
--------
If subjects formed *some* category structure -- not necessarily ours -- it should
appear in the rate at which different subjects co-assign two stimuli to the same
response.  We build the behavioural co-assignment matrix C, treat 1 - C as a
behavioural distance, cluster on it, and compare its geometry to several model
RDMs (label, Pearson, lagged xcorr, envelope, log-spectrum, MFCC).

Pipeline (per block):
    1. Build subject x stim matrix Y (NaN-aware).
    2. Co-assignment matrix C[i, j] = mean over subjects of 1[Y[s,i] == Y[s,j]],
       computed only where both are non-NaN.
    3. Behavioural RDM = 1 - C.
    4. Hierarchical clustering on RDM; report ARI vs experimenter labels for
       k = 2..5.
    5. RSA: vectorise upper-triangles of behavioural RDM and each model RDM;
       Spearman correlation; permutation p (shuffle stim indices on the model
       side).

Outputs
-------
outputs/book2/ch19-rsa/
    ch19_coassignment_<block>.npy
    ch19_clustering_<block>.csv
    ch19_rsa_<block>.csv
    ch19_coassignment_heatmap.png
    ch19_rsa_bars.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats        import spearmanr
from sklearn.cluster    import AgglomerativeClustering
from sklearn.metrics    import adjusted_rand_score, normalized_mutual_info_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import (load_all_trials, load_similarity_matrices,
                           build_subject_stim_matrix, METRICS)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book2" / "ch19-rsa"
OUT_PATH.mkdir(parents = True, exist_ok = True)

N_PERM = 2000
RNG    = np.random.default_rng(1)


# ============================================================================
# Helpers
# ============================================================================

def coassignment_matrix(Y):
    """C[i,j] = mean over subjects (with both responses present) of 1[Y[s,i]==Y[s,j]].
    Y has NaNs where the subject did not see that stimulus (shouldn't happen
    here, but we handle it).
    """
    n_sub, n_stim = Y.shape
    C = np.zeros((n_stim, n_stim))
    counts = np.zeros((n_stim, n_stim))
    valid = ~np.isnan(Y)
    for s in range(n_sub):
        v = valid[s]
        ys = Y[s]
        same = (ys[:, None] == ys[None, :]).astype(float)
        m = (v[:, None] & v[None, :]).astype(float)
        C += same * m
        counts += m
    C = np.where(counts > 0, C / np.maximum(counts, 1), 0.0)
    return C


def upper_tri(A):
    n = A.shape[0]
    iu = np.triu_indices(n, k = 1)
    return A[iu]


def perm_spearman(x_vec, y_vec, n_stim, n_perm, rng):
    """Permutation p-value of Spearman r between two RDM upper triangles by
    permuting the stimulus indices on the y side.  Returns observed r and p.
    """
    iu = np.triu_indices(n_stim, k = 1)
    r_obs = spearmanr(x_vec, y_vec).statistic
    # reconstruct the y RDM from y_vec
    Y = np.zeros((n_stim, n_stim))
    Y[iu] = y_vec
    Y = Y + Y.T
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n_stim)
        Yp = Y[np.ix_(p, p)]
        null[b] = spearmanr(x_vec, Yp[iu]).statistic
    p_two = (1 + np.sum(np.abs(null) >= np.abs(r_obs))) / (n_perm + 1)
    return r_obs, p_two


# ============================================================================
# Main
# ============================================================================

print("Loading data ...", flush = True)
trials   = load_all_trials()
sim_dict = load_similarity_matrices()

cluster_rows = []
rsa_rows     = []
heatmap_data = {}

for block in ("full_sentence", "imagined_sentence"):
    print(f"\n[{block}]", flush = True)
    Y, subjects, stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    n_stim = len(stim_nums)
    print(f"  Y shape = {Y.shape}", flush = True)

    C = coassignment_matrix(Y)
    np.save(OUT_PATH / f"ch19_coassignment_{block}.npy", C)
    behav_rdm = 1.0 - C
    np.fill_diagonal(behav_rdm, 0.0)

    # cluster on behavioural distance
    for k in (2, 3, 4, 5):
        clf = AgglomerativeClustering(n_clusters = k, metric = "precomputed",
                                      linkage = "average")
        clusters = clf.fit_predict(behav_rdm)
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        sizes = np.bincount(clusters).tolist()
        cluster_rows.append(dict(block_type = block, k = k,
                                 ari_vs_labels = ari, nmi_vs_labels = nmi,
                                 sizes = sizes))
        print(f"  k={k}  sizes={sizes}  ARI={ari:+.4f}  NMI={nmi:.4f}",
              flush = True)

    # RSA against model RDMs
    behav_vec = upper_tri(behav_rdm)
    label_rdm = (true_labels[:, None] != true_labels[None, :]).astype(float)
    np.fill_diagonal(label_rdm, 0.0)

    model_rdms = {"label": label_rdm}
    for metric, (sim, sim_stim_nums, sim_labels) in sim_dict.items():
        idx = {int(n): k for k, n in enumerate(sim_stim_nums)}
        block_idx = np.array([idx[int(n)] for n in stim_nums])
        sim_block = sim[np.ix_(block_idx, block_idx)]
        # convert similarity to dissimilarity in a comparable way:
        # for Pearson-style sims (range [-1,1]):  d = 1 - r
        d = 1.0 - sim_block
        np.fill_diagonal(d, 0.0)
        model_rdms[metric] = d

    for name, rdm in model_rdms.items():
        m_vec = upper_tri(rdm)
        r_obs, p_perm = perm_spearman(behav_vec, m_vec, n_stim, N_PERM, RNG)
        rsa_rows.append(dict(block_type = block, model = name,
                             spearman_r = r_obs, p_perm = p_perm))
        print(f"  RSA  behav vs {name:18s}  r={r_obs:+.4f}  p_perm={p_perm:.4f}",
              flush = True)

    heatmap_data[block] = (C, true_labels)

cluster_df = pd.DataFrame(cluster_rows)
cluster_df.to_csv(OUT_PATH / "ch19_clustering.csv", index = False)
rsa_df = pd.DataFrame(rsa_rows)
rsa_df.to_csv(OUT_PATH / "ch19_rsa.csv", index = False)


# ============================================================================
# Plots
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize = (14, 6))
for ax, (block, (C, lbl)) in zip(axes, heatmap_data.items()):
    order = np.argsort(lbl)  # group targets together, then distractors
    Cord  = C[np.ix_(order, order)]
    im = ax.imshow(Cord, cmap = "viridis", vmin = 0.3, vmax = 0.7,
                   aspect = "equal")
    n_t = int(lbl.sum())
    ax.axhline(n_t, color = "white", lw = 0.6)
    ax.axvline(n_t, color = "white", lw = 0.6)
    ax.set_title(f"{block}: co-assignment matrix\n(label-ordered, T then D)")
    ax.set_xlabel("Stimulus (label-ordered)")
    ax.set_ylabel("Stimulus (label-ordered)")
    plt.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04,
                 label = "P(same response)")
plt.tight_layout()
plt.savefig(OUT_PATH / "ch19_coassignment_heatmap.png", dpi = 130)
plt.close()

# RSA bar chart
fig, axes = plt.subplots(1, 2, figsize = (14, 5.5), sharey = True)
order_models = ["label"] + METRICS
for ax, blk in zip(axes, ["full_sentence", "imagined_sentence"]):
    sub = rsa_df[rsa_df["block_type"] == blk].set_index("model").loc[order_models]
    bars = ax.bar(range(len(order_models)), sub["spearman_r"].to_numpy(),
                  color = ["#444"] + ["#1f77b4"] * len(METRICS))
    for j, (m, r, p) in enumerate(zip(order_models, sub["spearman_r"], sub["p_perm"])):
        star = "*" if p < 0.05 else ""
        ax.text(j, r + 0.001 * np.sign(r) + 0.0005, f"p={p:.3f}{star}",
                ha = "center", fontsize = 7)
    ax.axhline(0, color = "k", lw = 0.6)
    ax.set_xticks(range(len(order_models)))
    ax.set_xticklabels(order_models, rotation = 30, ha = "right", fontsize = 8)
    ax.set_title(blk)
    ax.set_ylabel("Spearman r (behavioural RDM vs model RDM)")
plt.suptitle("RSA: which stimulus geometry best matches the response-derived geometry?",
             fontsize = 11)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch19_rsa_bars.png", dpi = 130)
plt.close()

print(f"\nChapter 19 done.  Outputs saved to: {OUT_PATH}", flush = True)
