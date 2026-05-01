"""Chapter 24.  Per-subject spectral logistic regression and template alignment.

For each (subject, block):
    1.  Compute magnitude FFT of all 150 stimuli (1481 bins) and bin into 32
        approximately log-spaced frequency bands.
    2.  Fit ridge logistic regression: subject_response ~ band features.
        Weight vector w_s lives in R^32.
    3.  Same regression on the experimenter labels gives w_true (block-level).
    4.  Per-subject alignment: cos(w_s, w_true).

Permutation null: shuffle the subject's response vector and refit.

Outputs
-------
outputs/book3/ch24-spectral-logistic/
    ch24_per_subject_spectral.csv
    ch24_alignment_hist.png
    ch24_weight_heatmap.png
    ch24_alignment_vs_dprime.png
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book3_helpers import (load_all_trials, load_block_waveforms,
                           build_subject_stim_matrix, magnitude_spectrum)

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch24-spectral-logistic"
OUT_PATH.mkdir(parents = True, exist_ok = True)

N_BANDS = 32
SR      = 8000
N_PERM  = 200
RIDGE_C = 0.5
RNG     = np.random.default_rng(0)


def log_band_features(spec, sr, n_bands):
    """Average magnitude spectrum into log-spaced bands.
    spec: (n_stim, F).  Returns (n_stim, n_bands)."""
    F = spec.shape[1]
    freqs = np.fft.rfftfreq(2 * (F - 1), d = 1.0 / sr)
    f_lo, f_hi = 50.0, sr / 2 - 50.0
    edges = np.geomspace(f_lo, f_hi, n_bands + 1)
    feats = np.zeros((spec.shape[0], n_bands))
    band_centers = []
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if mask.any():
            feats[:, i] = spec[:, mask].mean(axis = 1)
        band_centers.append(0.5 * (edges[i] + edges[i + 1]))
    return feats, np.array(band_centers)


def fit_w(X, y, C):
    """Ridge logistic, returns weight vector."""
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(penalty = "l2", C = C, solver = "liblinear",
                             max_iter = 2000)
    clf.fit(X, y)
    return clf.coef_.ravel()


def cos(a, b):
    if a is None or b is None:
        return np.nan
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# Main
# ============================================================================

trials = load_all_trials()

# Load Ch16 d' for the joint scatter
dprime_path = CUR_PATH / "outputs" / "book2" / "ch16-sdt" / "ch16_per_subject_sdt.csv"
if dprime_path.exists():
    dp_df = pd.read_csv(dprime_path)
    dp_df["subject"] = dp_df["subject"].astype(str)
else:
    dp_df = None

rows = []
weight_storage = {}    # for the heatmap

for block in ("full_sentence", "imagined_sentence"):
    print(f"\n[{block}]", flush = True)
    W, stim_nums, lbls, sr = load_block_waveforms(block)
    Spec  = magnitude_spectrum(W)
    feats, band_centers = log_band_features(Spec, sr, N_BANDS)
    Y, subjects, sub_stim_nums, true_labels = build_subject_stim_matrix(trials, block)
    idx = {int(n): k for k, n in enumerate(stim_nums)}
    order = np.array([idx[int(n)] for n in sub_stim_nums])
    feats_ord = feats[order]
    lbls_ord  = lbls[order]
    assert np.array_equal(lbls_ord, true_labels)

    scaler = StandardScaler().fit(feats_ord)
    X = scaler.transform(feats_ord)

    w_true = fit_w(X, lbls_ord, RIDGE_C)
    print(f"  ||w_true|| = {np.linalg.norm(w_true):.3f}", flush = True)

    block_weights = []
    block_aligns  = []
    block_subj    = []
    for s_idx, subj in enumerate(subjects):
        y = Y[s_idx]
        valid = ~np.isnan(y)
        Xv = X[valid]; yv = y[valid].astype(int)
        if len(np.unique(yv)) < 2:
            continue
        w_s = fit_w(Xv, yv, RIDGE_C)
        align = cos(w_s, w_true)

        # permutation null
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            yp = RNG.permutation(yv)
            wp = fit_w(Xv, yp, RIDGE_C)
            null[b] = cos(wp, w_true)
        p_two = (1 + np.sum(np.abs(null) >= np.abs(align))) / (N_PERM + 1)
        p_pos = (1 + np.sum(null >= align)) / (N_PERM + 1)

        rows.append(dict(subject = subj, block_type = block,
                         cos_w_true = align,
                         w_norm = float(np.linalg.norm(w_s)),
                         null_mean = float(null.mean()),
                         null_std  = float(null.std()),
                         p_perm_two = p_two,
                         p_perm_pos = p_pos,
                         n_yes = int((yv == 1).sum()),
                         n_no  = int((yv == 0).sum())))
        block_weights.append(w_s)
        block_aligns .append(align)
        block_subj   .append(subj)
    weight_storage[block] = (np.array(block_weights), np.array(block_aligns),
                             np.array(block_subj), w_true, band_centers)

per_subj = pd.DataFrame(rows)
per_subj.to_csv(OUT_PATH / "ch24_per_subject_spectral.csv", index = False)

print("\nSpectral alignment summary:", flush = True)
print(per_subj.groupby("block_type")["cos_w_true"].agg(
    ["mean", "median", "std", lambda x: (x > 0).mean()]), flush = True)
print("\n#subjects with p_perm_two < 0.05:", flush = True)
print(per_subj.groupby("block_type")["p_perm_two"].apply(lambda s: int((s < 0.05).sum())),
      flush = True)


# ============================================================================
# Plots
# ============================================================================

# Histogram of alignment per block
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
colors = {"full_sentence": "#1f77b4", "imagined_sentence": "#d62728"}
for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = per_subj[per_subj["block_type"] == blk]
    ax.hist(sub["cos_w_true"], bins = 18, color = colors[blk], edgecolor = "k",
            alpha = 0.85, label = "observed")
    ax.axvline(0, color = "k", lw = 0.7)
    ax.axvline(sub["cos_w_true"].mean(), color = "orange", lw = 2,
               label = f"mean = {sub['cos_w_true'].mean():+.3f}")
    ax.axvline(sub["null_mean"].mean(), color = "gray", lw = 1.5, ls = "--",
               label = f"mean(null) = {sub['null_mean'].mean():+.3f}")
    ax.set_xlabel("cos(w_subject, w_true)")
    ax.set_title(f"{blk}: per-subject spectral-logistic alignment")
    ax.legend(fontsize = 8)
plt.tight_layout()
plt.savefig(OUT_PATH / "ch24_alignment_hist.png", dpi = 130)
plt.close()

# Heatmap of subject weights, sorted by alignment, with w_true on top
for blk in ("full_sentence", "imagined_sentence"):
    Ws, aligns, subjs, w_true, band_centers = weight_storage[blk]
    ord_ = np.argsort(aligns)
    Ws_s = Ws[ord_]; subjs_s = subjs[ord_]; aligns_s = aligns[ord_]
    fig, axes = plt.subplots(2, 1, figsize = (12, 7),
                             gridspec_kw = dict(height_ratios = [1, 5]))
    ax_top, ax_bot = axes
    ax_top.bar(np.arange(N_BANDS), w_true, color = "k")
    ax_top.set_title(f"{blk}: w_true (top), per-subject w sorted by alignment (bottom)")
    ax_top.set_ylabel("w_true")
    ax_top.set_xticks([])
    vmax = np.percentile(np.abs(Ws_s), 95)
    im = ax_bot.imshow(Ws_s, aspect = "auto", cmap = "RdBu_r",
                       vmin = -vmax, vmax = vmax)
    ax_bot.set_yticks(np.arange(len(subjs_s)))
    ax_bot.set_yticklabels([f"{s[-3:]} ({a:+.2f})" for s, a in zip(subjs_s, aligns_s)],
                           fontsize = 7)
    xt = np.linspace(0, N_BANDS - 1, 6).astype(int)
    ax_bot.set_xticks(xt)
    ax_bot.set_xticklabels([f"{int(band_centers[i])}" for i in xt])
    ax_bot.set_xlabel("frequency band centre (Hz)")
    plt.colorbar(im, ax = ax_bot, fraction = 0.025, label = "weight")
    plt.tight_layout()
    plt.savefig(OUT_PATH / f"ch24_weight_heatmap_{blk}.png", dpi = 130)
    plt.close()

# Alignment vs d'
if dp_df is not None:
    per_subj["subject"] = per_subj["subject"].astype(str)
    merged = per_subj.merge(dp_df[["subject", "block_type", "d_prime"]],
                            on = ["subject", "block_type"], how = "left")
    fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
    for ax, blk in zip(axes, ("full_sentence", "imagined_sentence")):
        sub = merged[merged["block_type"] == blk]
        ax.scatter(sub["d_prime"], sub["cos_w_true"], color = colors[blk],
                   edgecolor = "k", s = 70, alpha = 0.85)
        ax.axhline(0, color = "gray", lw = 0.6)
        ax.axvline(0, color = "gray", lw = 0.6)
        ax.set_xlabel("d' (ch16)")
        ax.set_ylabel("cos(w_subject, w_true)")
        ax.set_title(blk)
    plt.suptitle("Per-subject spectral alignment vs SDT sensitivity",
                 fontsize = 11)
    plt.tight_layout()
    plt.savefig(OUT_PATH / "ch24_alignment_vs_dprime.png", dpi = 130)
    plt.close()

print(f"\nChapter 24 done.  Outputs saved to: {OUT_PATH}", flush = True)
