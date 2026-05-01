from pathlib import Path
import numpy as np
from scipy.io   import wavfile
from scipy.stats import pearsonr
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


def load_waveforms(directory):
    """Load all wav files in a directory as raw on-disk waveforms (float64).

    Args:
        directory: Path to directory containing wav files.

    Returns:
        Tuple (numbers_list, waveform_matrix) where waveform_matrix is
        shape (n_files, n_samples).
    """
    records = []
    nums    = []
    for wav_path in sorted(directory.glob("*.wav")):
        _, raw = wavfile.read(wav_path)
        data   = raw.astype(np.float64)
        records.append(data)
        nums.append(int(wav_path.stem))

    assert len(records) > 0, f"No wav files found in {directory}"
    return nums, np.vstack(records)


# ============================================================================
# Per-sample contribution analysis
# ============================================================================

def per_sample_pair_product_mean(mat_a, mat_b = None):
    """Mean per-sample product across all unique pairs.

    For centered rows, sum_t s_i[t] * s_j[t] = L * cov(s_i, s_j) (with ddof=0),
    so the mean across all pairs of s_i[t] * s_j[t] tells us how much sample t
    contributes (on average) to the pair's covariance.  Note: this is a
    covariance contribution, not a Pearson contribution; Pearson's denominator
    is the per-stimulus standard deviation, which is a property of each pair,
    not of any one sample.

    Args:
        mat_a: 2-D array (n_a x L) of centered waveforms.
        mat_b: Optional second 2-D array (n_b x L); if None, computes the
               within-group mean over upper-triangular pairs of mat_a.

    Returns:
        1-D array of length L with the mean per-sample pair product.
    """
    if mat_b is None:
        # Within-group: mean over i<j of s_i[t] * s_j[t].
        # Identity:  sum_{i<j} s_i s_j = 0.5 * (S^2 - sumsq)  where S = sum_i s_i.
        n     = mat_a.shape[0]
        S     = mat_a.sum(axis = 0)
        sumsq = (mat_a ** 2).sum(axis = 0)
        n_pairs   = n * (n - 1) / 2
        return 0.5 * (S ** 2 - sumsq) / n_pairs

    # Cross-group: mean over (i, j) of mat_a[i, t] * mat_b[j, t].
    sa = mat_a.sum(axis = 0)
    sb = mat_b.sum(axis = 0)
    return (sa * sb) / (mat_a.shape[0] * mat_b.shape[0])


def per_sample_agreement_fraction(mat):
    """Fraction of stimuli that share the same sign at each sample.

    Computed as (max(p+, p-)), where p+ and p- are the fractions of stimuli
    whose value at sample t is > 0 and < 0 respectively.  Range [0.5, 1.0],
    where 1.0 means perfect sign agreement and 0.5 means a coin flip.

    Args:
        mat: 2-D array (n x L) of z-normalized waveforms.

    Returns:
        1-D array of length L.
    """
    n          = mat.shape[0]
    pos_frac   = (mat > 0).sum(axis = 0) / n
    neg_frac   = (mat < 0).sum(axis = 0) / n
    return np.maximum(pos_frac, neg_frac)


# ============================================================================
# Main analysis
# ============================================================================

def main():
    """Per-sample pairwise contribution analysis for targets vs distractors."""
    cur_path     = Path(__file__).parent.resolve()
    stimuli_path = cur_path / ".." / "raw_data" / "audio_stimuli"
    output_path  = cur_path / "outputs" / "pearson" / "ch9-per-sample-contributions"
    output_path.mkdir(parents = True, exist_ok = True)

    # ---- Load all stimuli + template ---------------------------------------
    fs_targs_path = stimuli_path / "full_sentence"  / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"
    fs_dists_path = stimuli_path / "full_sentence"  / "distractors"
    is_dists_path = stimuli_path / "imagined_sentence" / "distractors"
    wall_path     = stimuli_path / "wall.wav"

    fs_t_nums, fs_t_mat = load_waveforms(fs_targs_path)
    is_t_nums, is_t_mat = load_waveforms(is_targs_path)
    fs_d_nums, fs_d_mat = load_waveforms(fs_dists_path)
    is_d_nums, is_d_mat = load_waveforms(is_dists_path)

    target_mat     = np.vstack([fs_t_mat, is_t_mat])
    distractor_mat = np.vstack([fs_d_mat, is_d_mat])

    assert wall_path.exists(), f"wall.wav not found at {wall_path}"
    sr, wall_raw = wavfile.read(wall_path)
    wall_data    = wall_raw.astype(np.float64)

    # Center every signal once (does not change Pearson r, simplifies per-sample math).
    target_mat     = target_mat     - target_mat.mean(axis     = 1, keepdims = True)
    distractor_mat = distractor_mat - distractor_mat.mean(axis = 1, keepdims = True)
    wall_c         = wall_data       - wall_data.mean()

    n_t, L = target_mat.shape
    n_d    = distractor_mat.shape[0]
    assert wall_c.shape[0] == L, f"Template length mismatch ({wall_c.shape[0]} vs {L})"

    print(f"Loaded {n_t} targets, {n_d} distractors, template L={L}, sr={sr} Hz.", flush = True)
    print(f"Using raw on-disk waveforms, centered (no z-normalization).", flush = True)

    # ---- Per-sample mean pair products -------------------------------------
    tt_contrib = per_sample_pair_product_mean(target_mat)
    dd_contrib = per_sample_pair_product_mean(distractor_mat)
    td_contrib = per_sample_pair_product_mean(target_mat, distractor_mat)

    # Sanity: sum over samples should be ~ proportional to mean pairwise r.
    # For z-normalized waves with ddof=0, sum_t s_i[t]*s_j[t] / L = Pearson r.
    mean_tt_r = np.mean(tt_contrib) * L / L   # = np.mean(tt_contrib)
    print(f"sum(tt_contrib)/L = {tt_contrib.sum() / L:+.5f}   "
          f"(expected ~ mean tt covariance, raw audio units)", flush = True)

    # ---- Per-sample sign agreement -----------------------------------------
    tt_sign_agree = per_sample_agreement_fraction(target_mat)
    dd_sign_agree = per_sample_agreement_fraction(distractor_mat)

    # ---- Per-sample variance across stimuli --------------------------------
    tt_var = target_mat.var(axis     = 0, ddof = 1)
    dd_var = distractor_mat.var(axis = 0, ddof = 1)

    # ---- Save the per-sample dataframe -------------------------------------
    times_sec = np.arange(L) / sr
    per_sample_df = pd.DataFrame({
        "sample_idx":      np.arange(L),
        "time_sec":        times_sec,
        "wall_c":          wall_c,
        "wall_energy":       wall_c ** 2,
        "tt_mean_product": tt_contrib,
        "dd_mean_product": dd_contrib,
        "td_mean_product": td_contrib,
        "tt_sign_agree":   tt_sign_agree,
        "dd_sign_agree":   dd_sign_agree,
        "tt_var_across":   tt_var,
        "dd_var_across":   dd_var,
    })
    per_sample_df.to_csv(output_path / "ch9b_per_sample.csv", index = False)

    # ---- How well does target agreement track the wall? --------------------
    # Compare per-sample tt contribution to wall_c^2 (i.e., wall energy).
    # Also against |wall| and against wall_c itself (signed).
    rows = []
    for label, x_signal, y_signal in [
        ("tt_contrib_vs_wall_sq",   tt_contrib, wall_c ** 2),
        ("tt_contrib_vs_abs_wall",  tt_contrib, np.abs(wall_c)),
        ("tt_contrib_vs_wall",      tt_contrib, wall_c),
        ("dd_contrib_vs_wall_sq",   dd_contrib, wall_c ** 2),
        ("dd_contrib_vs_abs_wall",  dd_contrib, np.abs(wall_c)),
        ("td_contrib_vs_wall_sq",   td_contrib, wall_c ** 2),
        ("tt_signagree_vs_abs_wall",tt_sign_agree, np.abs(wall_c)),
    ]:
        r, p = pearsonr(x_signal, y_signal)
        rows.append({"comparison": label, "pearson_r": float(r), "p_value": float(p)})
        print(f"{label:>30s}  r = {r:+.4f}  p = {fmt_p(p)}", flush = True)

    pd.DataFrame(rows).to_csv(output_path / "ch9b_signal_correlations.csv", index = False)

    # ---- Top-N most-contributing samples for tt ----------------------------
    top_n        = 50
    top_idx      = np.argsort(tt_contrib)[::-1][:top_n]
    top_df       = pd.DataFrame({
        "rank":             np.arange(1, top_n + 1),
        "sample_idx":       top_idx,
        "time_sec":         times_sec[top_idx],
        "tt_mean_product":  tt_contrib[top_idx],
        "wall_c":           wall_c[top_idx],
        "tt_sign_agree":    tt_sign_agree[top_idx],
    })
    top_df.to_csv(output_path / "ch9b_top_contributing_samples.csv", index = False)

    # =======================================================================
    # PLOTS
    # =======================================================================

    # ---- Plot 1: per-sample tt contribution overlaid on wall_c -------------
    fig, axes = plt.subplots(3, 1, figsize = (12, 8), sharex = True)

    ax = axes[0]
    ax.plot(times_sec, wall_c, color = "black", linewidth = 0.7)
    ax.set_ylabel("Wall (z)")
    ax.set_title("Wall Template")
    ax.grid(alpha = 0.3)

    ax = axes[1]
    ax.plot(times_sec, tt_contrib, color = "tomato",     linewidth = 0.7, label = "target-target")
    ax.plot(times_sec, dd_contrib, color = "steelblue",  linewidth = 0.7, label = "distractor-distractor", alpha = 0.7)
    ax.plot(times_sec, td_contrib, color = "darkorange", linewidth = 0.7, label = "target-distractor",     alpha = 0.7)
    ax.axhline(0, color = "gray", linewidth = 0.6, linestyle = "--")
    ax.set_ylabel("Mean per-sample product\n(across all pairs)")
    ax.set_title("Per-Sample Pairwise Contribution")
    ax.legend(fontsize = 8, loc = "upper right")
    ax.grid(alpha = 0.3)

    ax = axes[2]
    ax.plot(times_sec, wall_c ** 2, color = "black", linewidth = 0.7, label = "wall_c^2 (energy)")
    ax2 = ax.twinx()
    ax2.plot(times_sec, tt_contrib, color = "tomato", linewidth = 0.7, alpha = 0.85, label = "tt contribution")
    ax.set_ylabel("Wall energy (wall_c^2)")
    ax2.set_ylabel("tt contribution", color = "tomato")
    ax.set_xlabel("Time (s)")
    ax.set_title("Wall Energy vs Target-Target Contribution (Twin Axes)")
    ax.grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9b_per_sample_overlay.png", dpi = 200)
    plt.close(fig)

    # ---- Plot 2: scatter, per-sample tt contribution vs wall energy --------
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    r1, p1 = pearsonr(tt_contrib, wall_c ** 2)
    axes[0].scatter(wall_c ** 2, tt_contrib, s = 4, alpha = 0.4, color = "tomato")
    axes[0].axhline(0, color = "gray", linewidth = 0.6, linestyle = "--")
    axes[0].set_xlabel("Wall energy at sample (wall_c^2)")
    axes[0].set_ylabel("tt mean product at sample")
    axes[0].set_title(f"Per-Sample tt Contribution vs Wall Energy\n  r = {r1:+.3f}  p = {fmt_p(p1)}")
    axes[0].grid(alpha = 0.3)

    r2, p2 = pearsonr(tt_contrib, wall_c)
    axes[1].scatter(wall_c, tt_contrib, s = 4, alpha = 0.4, color = "tomato")
    axes[1].axhline(0, color = "gray", linewidth = 0.6, linestyle = "--")
    axes[1].axvline(0, color = "gray", linewidth = 0.6, linestyle = "--")
    axes[1].set_xlabel("Wall amplitude (wall_c)")
    axes[1].set_ylabel("tt mean product at sample")
    axes[1].set_title(f"Per-Sample tt Contribution vs Wall Amplitude\n  r = {r2:+.3f}  p = {fmt_p(p2)}")
    axes[1].grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9b_scatter_vs_wall.png", dpi = 200)
    plt.close(fig)

    # ---- Plot 3: sign-agreement profile -----------------------------------
    fig, axes = plt.subplots(2, 1, figsize = (12, 6), sharex = True)

    axes[0].plot(times_sec, tt_sign_agree, color = "tomato",    linewidth = 0.6, label = "targets")
    axes[0].plot(times_sec, dd_sign_agree, color = "steelblue", linewidth = 0.6, label = "distractors", alpha = 0.7)
    axes[0].axhline(0.5, color = "gray", linewidth = 0.6, linestyle = "--", label = "chance (0.5)")
    axes[0].set_ylabel("Fraction sharing sign\n(majority side)")
    axes[0].set_title("Per-Sample Sign Agreement Across Stimuli")
    axes[0].legend(fontsize = 8)
    axes[0].grid(alpha = 0.3)

    axes[1].plot(times_sec, np.abs(wall_c), color = "black", linewidth = 0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("|wall|")
    axes[1].set_title("Wall Magnitude (for visual alignment)")
    axes[1].grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9b_sign_agreement.png", dpi = 200)
    plt.close(fig)

    # ---- Plot 4: per-sample variance (low variance = consensus) ------------
    fig, ax = plt.subplots(figsize = (12, 4))
    ax.plot(times_sec, tt_var, color = "tomato",    linewidth = 0.6, label = "targets")
    ax.plot(times_sec, dd_var, color = "steelblue", linewidth = 0.6, label = "distractors", alpha = 0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Variance across stimuli\n(z-normalized)")
    ax.set_title("Per-Sample Variance Across Stimuli (lower = more consensus)")
    ax.legend(fontsize = 8)
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ch9b_per_sample_variance.png", dpi = 200)
    plt.close(fig)

    # ---- Plot 5: top-N contributing samples marked on wall -----------------
    fig, ax = plt.subplots(figsize = (12, 4))
    ax.plot(times_sec, wall_c, color = "black", linewidth = 0.7, alpha = 0.7, label = "wall_c")
    ax.scatter(times_sec[top_idx], wall_c[top_idx], s = 25, color = "tomato",
               zorder = 3, label = f"Top {top_n} tt-contributing samples")
    ax.axhline(0, color = "gray", linewidth = 0.6, linestyle = "--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wall (z)")
    ax.set_title(f"Top {top_n} Target-Target Contributing Samples Overlaid on Wall")
    ax.legend(fontsize = 9)
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ch9b_top_samples_on_wall.png", dpi = 200)
    plt.close(fig)

    print("Per-sample contribution analysis done.  Outputs in:", output_path, flush = True)


if __name__ == "__main__":
    main()
