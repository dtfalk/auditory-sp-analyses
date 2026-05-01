from pathlib import Path
import numpy as np
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def fmt_p(p_val):
    """Format a p-value as decimal unless it is smaller than 1e-4.

    Args:
        p_val: Float p-value.

    Returns:
        Formatted string.
    """
    if p_val < 1e-4:
        return f"{p_val:.3e}"
    return f"{p_val:.4f}"


def pairwise_upper_tri(waveform_matrix):
    """Upper-triangle Pearson r values from a set of waveforms.

    Args:
        waveform_matrix: 2-D array (n_stimuli x n_samples).

    Returns:
        1-D array of pairwise Pearson r values (k=1 upper triangle).
    """
    corr = np.corrcoef(waveform_matrix)
    idx  = np.triu_indices_from(corr, k = 1)
    return corr[idx]


def cross_pairwise(mat_a, mat_b):
    """All pairwise Pearson r values between every row of mat_a and every row of mat_b.

    Args:
        mat_a: 2-D array (n_a x n_samples).
        mat_b: 2-D array (n_b x n_samples).

    Returns:
        1-D array of n_a * n_b Pearson r values.
    """
    stacked = np.vstack([mat_a, mat_b])
    full    = np.corrcoef(stacked)
    n_a     = mat_a.shape[0]
    block   = full[:n_a, n_a:]
    return block.ravel()


def percentile_rank(observed, null_samples):
    """Fraction of null samples <= observed (in [0, 1]).

    Args:
        observed:     Scalar observed value.
        null_samples: 1-D array of null statistic values.

    Returns:
        Float percentile rank.
    """
    return float(np.mean(null_samples <= observed))


def two_sided_p(observed, null_samples):
    """Two-sided permutation p-value: P(|null| >= |observed|), with +1 smoothing.

    Args:
        observed:     Scalar observed statistic.
        null_samples: 1-D array of null statistic values (assumed centered ~ 0).

    Returns:
        Float p-value in (0, 1].
    """
    n_extreme = int(np.sum(np.abs(null_samples) >= abs(observed)))
    return (n_extreme + 1) / (len(null_samples) + 1)


def one_sided_p_greater(observed, null_samples):
    """One-sided p-value: P(null >= observed), with +1 smoothing.

    Args:
        observed:     Scalar observed statistic.
        null_samples: 1-D array of null statistic values.

    Returns:
        Float p-value in (0, 1].
    """
    n_ge = int(np.sum(null_samples >= observed))
    return (n_ge + 1) / (len(null_samples) + 1)


# ============================================================================
# Data Loading
# ============================================================================

def load_stimuli(cur_path):
    """Load all target and distractor waveforms (raw on-disk PCM, centered).

    Args:
        cur_path: Path to the script directory.

    Returns:
        Tuple (target_mat, distractor_mat), each centered (n x L) float64.
    """
    stimuli_path  = cur_path / ".." / "raw_data" / "audio_stimuli"
    fs_targs_path = stimuli_path / "full_sentence"     / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"
    fs_dists_path = stimuli_path / "full_sentence"     / "distractors"
    is_dists_path = stimuli_path / "imagined_sentence" / "distractors"

    def read_dir(directory):
        rows = []
        for wav_path in sorted(directory.glob("*.wav")):
            _, raw = wavfile.read(wav_path)
            rows.append(raw.astype(np.float64))
        assert len(rows) > 0, f"No wav files in {directory}"
        return np.vstack(rows)

    target_mat     = np.vstack([read_dir(fs_targs_path), read_dir(is_targs_path)])
    distractor_mat = np.vstack([read_dir(fs_dists_path), read_dir(is_dists_path)])

    # Center every signal (does not change Pearson r).
    target_mat     = target_mat     - target_mat.mean(axis     = 1, keepdims = True)
    distractor_mat = distractor_mat - distractor_mat.mean(axis = 1, keepdims = True)

    print(f"Loaded {target_mat.shape[0]} targets, {distractor_mat.shape[0]} distractors, "
          f"length L={target_mat.shape[1]} samples (raw on-disk, centered).", flush = True)
    return target_mat, distractor_mat


# ============================================================================
# Null Models
# ============================================================================

def white_noise_null_mean_r(n_stimuli, L, n_reps, rng):
    """Null distribution of mean upper-tri Pearson r for a set of n Gaussian-noise clips.

    Each replicate draws an (n_stimuli x L) matrix of i.i.d. N(0,1) samples and
    computes the mean of its n*(n-1)/2 upper-triangular pairwise Pearson r values.
    Pearson r is scale invariant, so Gaussian vs other unit-energy distributions
    give the same null to leading order.

    Args:
        n_stimuli: Number of clips per replicate.
        L:         Length of each clip in samples.
        n_reps:    Number of replicates.
        rng:       numpy Generator.

    Returns:
        1-D array of length n_reps with the per-replicate mean Pearson r.
    """
    out = np.empty(n_reps, dtype = np.float64)
    for i in range(n_reps):
        mat       = rng.standard_normal((n_stimuli, L))
        out[i]    = pairwise_upper_tri(mat).mean()
        if (i + 1) % 100 == 0:
            print(f"    white-noise null  n={n_stimuli}  rep {i + 1}/{n_reps}", flush = True)
    return out


def label_permutation_null(pooled_mat, n_a, n_reps, rng):
    """Null distribution under random relabeling of pooled stimuli into two groups.

    For each replicate, pooled stimuli are randomly partitioned into a "group A"
    of size n_a and a "group B" of size n_b = pooled_mat.shape[0] - n_a, then
    mean within-A, within-B, and across (A x B) pairwise Pearson r are recorded.

    Computes the full pooled correlation matrix once and reuses it across
    replicates by re-indexing — far cheaper than re-correlating each rep.

    Args:
        pooled_mat: (n_total x L) matrix of stimuli (pre-centered).
        n_a:        Size of group A in each replicate.
        n_reps:     Number of replicates.
        rng:        numpy Generator.

    Returns:
        Dict with keys 'mean_aa', 'mean_bb', 'mean_ab' — each a 1-D array of length n_reps.
    """
    n_total = pooled_mat.shape[0]
    full_r  = np.corrcoef(pooled_mat)               # (n_total x n_total)

    mean_aa = np.empty(n_reps, dtype = np.float64)
    mean_bb = np.empty(n_reps, dtype = np.float64)
    mean_ab = np.empty(n_reps, dtype = np.float64)

    for i in range(n_reps):
        perm    = rng.permutation(n_total)
        a_idx   = perm[:n_a]
        b_idx   = perm[n_a:]

        aa_block = full_r[np.ix_(a_idx, a_idx)]
        bb_block = full_r[np.ix_(b_idx, b_idx)]
        ab_block = full_r[np.ix_(a_idx, b_idx)]

        # Within-group: average upper triangle (k=1) only.
        triu_aa = aa_block[np.triu_indices_from(aa_block, k = 1)]
        triu_bb = bb_block[np.triu_indices_from(bb_block, k = 1)]

        mean_aa[i] = triu_aa.mean()
        mean_bb[i] = triu_bb.mean()
        mean_ab[i] = ab_block.mean()

        if (i + 1) % 500 == 0:
            print(f"    label-permutation null  rep {i + 1}/{n_reps}", flush = True)

    return {"mean_aa": mean_aa, "mean_bb": mean_bb, "mean_ab": mean_ab}


# ============================================================================
# Chapter 10
# ============================================================================

def chapter_10(target_mat, distractor_mat, output_path,
               n_white_reps = 500, n_perm_reps = 5000, seed = 0xC0FFEE):
    """Null models and permutation tests for Pearson structure.

    Two complementary nulls:
      A. White-noise null — mean within-group Pearson r expected from n random
         clips of length L with no shared structure. Localizes whether the
         observed within-target cohesion (and within-distractor structure) sits
         outside the white-noise envelope.
      B. Label-permutation null — pool targets + distractors and randomly
         relabel into two groups of the same sizes. The observed separation
         (mean_tt - mean_dd, and mean_tt - mean_td) is compared to the
         distribution of the same statistics under random relabeling.

    Args:
        target_mat:     (n_t x L) centered target waveforms.
        distractor_mat: (n_d x L) centered distractor waveforms.
        output_path:    Path to save outputs.
        n_white_reps:   Number of white-noise null replicates per group size.
        n_perm_reps:    Number of label-permutation replicates.
        seed:           RNG seed for reproducibility.

    Returns:
        None
    """
    output_path.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(seed)

    n_t, L = target_mat.shape
    n_d    = distractor_mat.shape[0]

    # -----------------------------------------------------------------------
    # Step 1: Observed statistics.
    # -----------------------------------------------------------------------
    tt_obs    = pairwise_upper_tri(target_mat)
    dd_obs    = pairwise_upper_tri(distractor_mat)
    td_obs    = cross_pairwise(target_mat, distractor_mat)

    mean_tt   = float(tt_obs.mean())
    mean_dd   = float(dd_obs.mean())
    mean_td   = float(td_obs.mean())

    sep_tt_dd = mean_tt - mean_dd
    sep_tt_td = mean_tt - mean_td

    print(f"Observed: mean_tt={mean_tt:+.5f}  mean_dd={mean_dd:+.5f}  mean_td={mean_td:+.5f}",
          flush = True)
    print(f"Observed separations: tt-dd={sep_tt_dd:+.5f}  tt-td={sep_tt_td:+.5f}", flush = True)

    # -----------------------------------------------------------------------
    # Step 2: White-noise null — per-group mean within-group Pearson r.
    # -----------------------------------------------------------------------
    print(f"Step 2: white-noise null  ({n_white_reps} reps each, n_t={n_t}, n_d={n_d}, L={L}) ...",
          flush = True)
    wn_null_t = white_noise_null_mean_r(n_t, L, n_white_reps, rng)
    wn_null_d = white_noise_null_mean_r(n_d, L, n_white_reps, rng)

    # Two-sided p (centered around 0): how extreme is the observed mean?
    p_wn_tt   = two_sided_p(mean_tt, wn_null_t)
    p_wn_dd   = two_sided_p(mean_dd, wn_null_d)
    pct_wn_tt = percentile_rank(mean_tt, wn_null_t)
    pct_wn_dd = percentile_rank(mean_dd, wn_null_d)

    print(f"  white-noise null mean_tt: null mean={wn_null_t.mean():+.6f} std={wn_null_t.std():.6f} "
          f"min={wn_null_t.min():+.6f} max={wn_null_t.max():+.6f}", flush = True)
    print(f"    observed mean_tt={mean_tt:+.5f}  percentile={pct_wn_tt * 100:.2f}%  "
          f"two-sided p={fmt_p(p_wn_tt)}", flush = True)
    print(f"  white-noise null mean_dd: null mean={wn_null_d.mean():+.6f} std={wn_null_d.std():.6f} "
          f"min={wn_null_d.min():+.6f} max={wn_null_d.max():+.6f}", flush = True)
    print(f"    observed mean_dd={mean_dd:+.5f}  percentile={pct_wn_dd * 100:.2f}%  "
          f"two-sided p={fmt_p(p_wn_dd)}", flush = True)

    pd.DataFrame({
        "rep":          np.arange(n_white_reps),
        "wn_null_mean_tt": wn_null_t,
        "wn_null_mean_dd": wn_null_d,
    }).to_csv(output_path / "ch10_white_noise_null.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 3: Label-permutation null — pool and reshuffle group labels.
    # -----------------------------------------------------------------------
    print(f"Step 3: label-permutation null  ({n_perm_reps} reps, pool size {n_t + n_d}) ...",
          flush = True)
    pooled = np.vstack([target_mat, distractor_mat])
    perm   = label_permutation_null(pooled, n_a = n_t, n_reps = n_perm_reps, rng = rng)

    sep_aa_bb_null = perm["mean_aa"] - perm["mean_bb"]
    sep_aa_ab_null = perm["mean_aa"] - perm["mean_ab"]

    p_perm_tt_dd   = two_sided_p(sep_tt_dd, sep_aa_bb_null)
    p_perm_tt_td   = two_sided_p(sep_tt_td, sep_aa_ab_null)
    p_perm_tt      = one_sided_p_greater(mean_tt, perm["mean_aa"])  # tt above relabeled-A mean?
    p_perm_dd_low  = (np.sum(perm["mean_bb"] <= mean_dd) + 1) / (n_perm_reps + 1)

    print(f"  permutation null mean_aa: mean={perm['mean_aa'].mean():+.5f} "
          f"std={perm['mean_aa'].std():.5f}  observed mean_tt={mean_tt:+.5f}  "
          f"one-sided p(>= obs)={fmt_p(p_perm_tt)}", flush = True)
    print(f"  permutation null mean_bb: mean={perm['mean_bb'].mean():+.5f} "
          f"std={perm['mean_bb'].std():.5f}  observed mean_dd={mean_dd:+.5f}  "
          f"one-sided p(<= obs)={fmt_p(p_perm_dd_low)}", flush = True)
    print(f"  separation tt-dd: null mean={sep_aa_bb_null.mean():+.5f} "
          f"std={sep_aa_bb_null.std():.5f}  observed={sep_tt_dd:+.5f}  "
          f"two-sided p={fmt_p(p_perm_tt_dd)}", flush = True)
    print(f"  separation tt-td: null mean={sep_aa_ab_null.mean():+.5f} "
          f"std={sep_aa_ab_null.std():.5f}  observed={sep_tt_td:+.5f}  "
          f"two-sided p={fmt_p(p_perm_tt_td)}", flush = True)

    pd.DataFrame({
        "rep":            np.arange(n_perm_reps),
        "perm_mean_aa":   perm["mean_aa"],
        "perm_mean_bb":   perm["mean_bb"],
        "perm_mean_ab":   perm["mean_ab"],
        "perm_sep_aa_bb": sep_aa_bb_null,
        "perm_sep_aa_ab": sep_aa_ab_null,
    }).to_csv(output_path / "ch10_label_permutation_null.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 4: Summary table.
    # -----------------------------------------------------------------------
    summary_rows = [
        {"test":     "white_noise_null_mean_tt",
         "observed":  mean_tt,
         "null_mean": float(wn_null_t.mean()),
         "null_std":  float(wn_null_t.std()),
         "percentile":pct_wn_tt,
         "p_value":   p_wn_tt,
         "tail":      "two-sided"},
        {"test":     "white_noise_null_mean_dd",
         "observed":  mean_dd,
         "null_mean": float(wn_null_d.mean()),
         "null_std":  float(wn_null_d.std()),
         "percentile":pct_wn_dd,
         "p_value":   p_wn_dd,
         "tail":      "two-sided"},
        {"test":     "label_perm_mean_tt_above_relabeled",
         "observed":  mean_tt,
         "null_mean": float(perm["mean_aa"].mean()),
         "null_std":  float(perm["mean_aa"].std()),
         "percentile":percentile_rank(mean_tt, perm["mean_aa"]),
         "p_value":   p_perm_tt,
         "tail":      "one-sided (>=)"},
        {"test":     "label_perm_mean_dd_below_relabeled",
         "observed":  mean_dd,
         "null_mean": float(perm["mean_bb"].mean()),
         "null_std":  float(perm["mean_bb"].std()),
         "percentile":percentile_rank(mean_dd, perm["mean_bb"]),
         "p_value":   p_perm_dd_low,
         "tail":      "one-sided (<=)"},
        {"test":     "label_perm_separation_tt_minus_dd",
         "observed":  sep_tt_dd,
         "null_mean": float(sep_aa_bb_null.mean()),
         "null_std":  float(sep_aa_bb_null.std()),
         "percentile":percentile_rank(sep_tt_dd, sep_aa_bb_null),
         "p_value":   p_perm_tt_dd,
         "tail":      "two-sided"},
        {"test":     "label_perm_separation_tt_minus_td",
         "observed":  sep_tt_td,
         "null_mean": float(sep_aa_ab_null.mean()),
         "null_std":  float(sep_aa_ab_null.std()),
         "percentile":percentile_rank(sep_tt_td, sep_aa_ab_null),
         "p_value":   p_perm_tt_td,
         "tail":      "two-sided"},
    ]
    pd.DataFrame(summary_rows).to_csv(output_path / "ch10_summary.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 5: Plots.
    # -----------------------------------------------------------------------
    # Plot 1: white-noise null histograms with observed markers.
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    ax = axes[0]
    ax.hist(wn_null_t, bins = 40, color = "lightgray", edgecolor = "black")
    ax.axvline(mean_tt, color = "tomato", linewidth = 2,
               label = f"observed mean_tt = {mean_tt:+.5f}")
    ax.set_title(f"White-noise null: mean within-group r (n={n_t}, L={L})")
    ax.set_xlabel("Mean upper-tri Pearson r")
    ax.set_ylabel("Frequency")
    ax.legend()

    ax = axes[1]
    ax.hist(wn_null_d, bins = 40, color = "lightgray", edgecolor = "black")
    ax.axvline(mean_dd, color = "steelblue", linewidth = 2,
               label = f"observed mean_dd = {mean_dd:+.5f}")
    ax.set_title(f"White-noise null: mean within-group r (n={n_d}, L={L})")
    ax.set_xlabel("Mean upper-tri Pearson r")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path / "ch10_white_noise_null.png", dpi = 150)
    plt.close(fig)

    # Plot 2: label-permutation nulls — group means vs observed.
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    ax = axes[0]
    ax.hist(perm["mean_aa"], bins = 40, color = "lightgray", edgecolor = "black",
            alpha = 0.85, label = "perm: relabeled group A (size n_t)")
    ax.hist(perm["mean_bb"], bins = 40, color = "lightsteelblue", edgecolor = "black",
            alpha = 0.55, label = "perm: relabeled group B (size n_d)")
    ax.axvline(mean_tt, color = "tomato", linewidth = 2,
               label = f"observed mean_tt = {mean_tt:+.5f}")
    ax.axvline(mean_dd, color = "steelblue", linewidth = 2,
               label = f"observed mean_dd = {mean_dd:+.5f}")
    ax.set_title("Label-permutation null: within-group means")
    ax.set_xlabel("Mean upper-tri Pearson r")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize = 8)

    ax = axes[1]
    ax.hist(sep_aa_bb_null, bins = 40, color = "lightgray", edgecolor = "black",
            alpha = 0.85, label = "perm: mean_A - mean_B")
    ax.hist(sep_aa_ab_null, bins = 40, color = "lightsteelblue", edgecolor = "black",
            alpha = 0.55, label = "perm: mean_A - mean_AB")
    ax.axvline(sep_tt_dd, color = "tomato", linewidth = 2,
               label = f"observed tt-dd = {sep_tt_dd:+.5f}")
    ax.axvline(sep_tt_td, color = "darkorange", linewidth = 2, linestyle = "--",
               label = f"observed tt-td = {sep_tt_td:+.5f}")
    ax.set_title("Label-permutation null: group separation")
    ax.set_xlabel("Mean-r difference")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize = 8)

    fig.tight_layout()
    fig.savefig(output_path / "ch10_label_permutation_null.png", dpi = 150)
    plt.close(fig)

    print(f"Chapter 10 done.  Outputs saved to: {output_path}", flush = True)


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path    = Path(__file__).resolve().parent
    output_path = cur_path / "outputs" / "pearson" / "ch10-null-models"

    target_mat, distractor_mat = load_stimuli(cur_path)
    chapter_10(target_mat, distractor_mat, output_path)


if __name__ == "__main__":
    main()
