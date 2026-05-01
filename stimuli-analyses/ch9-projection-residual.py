from pathlib import Path
import numpy as np
from scipy.io  import wavfile
from scipy.stats import ttest_ind, ttest_1samp, wilcoxon, pearsonr
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
    block   = full[:n_a, n_a:]          # off-diagonal block
    return block.ravel()


def condition_summary(vals, label):
    """Build a one-row dict of summary stats for a 1-D array of Pearson r values.

    Args:
        vals:  1-D numpy array of Pearson r values.
        label: String label for the condition.

    Returns:
        Dict with condition, mean_r, std_r, se_r, n_pairs.
    """
    return {
        "condition": label,
        "mean_r":    float(np.mean(vals)),
        "std_r":     float(np.std(vals, ddof = 1)),
        "se_r":      float(np.std(vals, ddof = 1) / np.sqrt(len(vals))),
        "n_pairs":   int(len(vals)),
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_stimuli(cur_path):
    """Load and z-normalize all target and distractor waveforms.

    Args:
        cur_path: Path to the script directory.

    Returns:
        Tuple (targets_df, distractors_df), each a DataFrame with columns
        [stimulus_number, waveform].
    """
    stimuli_path  = cur_path / ".." / "raw_data" / "audio_stimuli"
    fs_targs_path = stimuli_path / "full_sentence"  / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"
    fs_dists_path = stimuli_path / "full_sentence"  / "distractors"
    is_dists_path = stimuli_path / "imagined_sentence" / "distractors"
    wall_path     = stimuli_path / "wall.wav"

    def read_wavs(directory):
        records = []
        for wav_path in sorted(directory.glob("*.wav")):
            _, raw = wavfile.read(wav_path)
            data   = raw.astype(np.float64)
            records.append({
                "stimulus_number": int(wav_path.stem),
                "waveform":        data,
            })
        return records

    target_records     = read_wavs(fs_targs_path) + read_wavs(is_targs_path)
    distractor_records = read_wavs(fs_dists_path) + read_wavs(is_dists_path)

    assert len(target_records)     > 0, "No target wavs found — check path"
    assert len(distractor_records) > 0, "No distractor wavs found — check path"
    assert wall_path.exists(),           f"wall.wav not found at {wall_path}"

    _, wall_raw  = wavfile.read(wall_path)
    wall_wav     = wall_raw.astype(np.float64)

    targets_df     = pd.DataFrame(target_records)
    distractors_df = pd.DataFrame(distractor_records)

    print(f"Loaded {len(targets_df)} targets, {len(distractors_df)} distractors.", flush = True)
    print(f"Template (wall.wav, raw on-disk): length = {len(wall_wav)} samples.", flush = True)
    return targets_df, distractors_df, wall_wav


# ============================================================================
# Chapter 9
# ============================================================================

def chapter_9(targets_df, distractors_df, wall_wav, output_path):
    """Projection onto the template and residual pairwise structure.

    Separates 'wall-like' variance from other variance by:
      1. Using wall.wav (the original template) as the projection axis.
      2. Computing each stimulus's projection coefficient onto the template.
      3. Residualizing (removing the template component) and re-z-normalizing.
      4. Repeating pairwise Pearson (tt/dd/td) on residuals.
      5. Comparing before/after cohesion to answer whether target cohesion is
         entirely driven by shared template alignment.

    Args:
        targets_df:     DataFrame with columns [stimulus_number, waveform].
        distractors_df: DataFrame with columns [stimulus_number, waveform].
        wall_wav:       Z-normalized wall.wav waveform (1-D numpy array).
        output_path:    Path to save all outputs.

    Returns:
        None
    """
    output_path.mkdir(parents = True, exist_ok = True)

    # -----------------------------------------------------------------------
    # Step 1: Use wall.wav (raw on-disk waveform) as the template.
    #   The original pipeline computed r_scores on raw PCM_16, then RMS-normalized
    #   in-place after.  Pearson r is centering/scale invariant, so r values are
    #   identical on raw vs RMS-normalized signals.
    #
    #   We center every signal once up front (does not change Pearson r) so the
    #   OLS projection geometry is clean:
    #     proj_coeff(s) = dot(s_c, t_c) / dot(t_c, t_c)
    #     residual      = s_c - proj_coeff * t_c     (orthogonal to t_c)
    # -----------------------------------------------------------------------
    target_mat     = np.vstack(targets_df["waveform"].to_list())      # (n_t x L)
    distractor_mat = np.vstack(distractors_df["waveform"].to_list())  # (n_d x L)

    template_raw = wall_wav
    template     = template_raw - np.mean(template_raw)               # centered
    t_dot_t      = float(np.dot(template, template))
    assert t_dot_t > 0, "Template has zero variance"
    assert template.shape[0] == target_mat.shape[1], (
        f"Template length {template.shape[0]} != stimulus length {target_mat.shape[1]}"
    )

    # Center every stimulus (does not change Pearson r).
    target_mat     = target_mat     - target_mat.mean(axis     = 1, keepdims = True)
    distractor_mat = distractor_mat - distractor_mat.mean(axis = 1, keepdims = True)

    # -----------------------------------------------------------------------
    # Step 2: Projection coefficients (OLS regression slope onto template).
    # -----------------------------------------------------------------------
    L = template.shape[0]

    t_proj = (target_mat     @ template) / t_dot_t   # shape (n_t,)
    d_proj = (distractor_mat @ template) / t_dot_t   # shape (n_d,)

    proj_rows = (
        [{"stimulus_number": int(r["stimulus_number"]),
          "label":           "target",
          "projection_coeff": float(t_proj[i])}
         for i, (_, r) in enumerate(targets_df.iterrows())]
        +
        [{"stimulus_number": int(r["stimulus_number"]),
          "label":           "distractor",
          "projection_coeff": float(d_proj[i])}
         for i, (_, r) in enumerate(distractors_df.iterrows())]
    )
    proj_df = pd.DataFrame(proj_rows)
    proj_df.to_csv(output_path / "ch9_projection_coefficients.csv", index = False)

    # Pearson r-to-template, computed independently for interpretability.
    t_pearson_r = np.array([pearsonr(s, template)[0] for s in target_mat])
    d_pearson_r = np.array([pearsonr(s, template)[0] for s in distractor_mat])
    pd.DataFrame({
        "stimulus_number":       (targets_df["stimulus_number"].tolist()
                                  + distractors_df["stimulus_number"].tolist()),
        "label":                 ["target"] * len(targets_df) + ["distractor"] * len(distractors_df),
        "pearson_r_to_template": np.concatenate([t_pearson_r, d_pearson_r]),
    }).to_csv(output_path / "ch9_pearson_r_to_template.csv", index = False)

    print(f"Targets    - mean OLS proj coeff: {t_proj.mean():+.5f}  "
          f"std: {t_proj.std():.5f}  |  mean Pearson r to wall: {t_pearson_r.mean():+.5f}", flush = True)
    print(f"Distractors- mean OLS proj coeff: {d_proj.mean():+.5f}  "
          f"std: {d_proj.std():.5f}  |  mean Pearson r to wall: {d_pearson_r.mean():+.5f}", flush = True)

    # -----------------------------------------------------------------------
    # Step 3: Residualize.
    #   residual = s_c - proj_coeff * t_c   (orthogonal to template by construction)
    #   No re-normalization: Pearson r between residuals is scale-invariant.
    # -----------------------------------------------------------------------
    def residualize(mat, proj_coeffs):
        """Remove the OLS template component, leaving the orthogonal residual."""
        residuals = mat - proj_coeffs[:, np.newaxis] * template[np.newaxis, :]
        stds = residuals.std(axis = 1)
        assert np.all(stds > 0), "Zero-variance residual - template explains 100% of a stimulus"
        return residuals

    target_resid     = residualize(target_mat,     t_proj)
    distractor_resid = residualize(distractor_mat, d_proj)

    # -----------------------------------------------------------------------
    # Step 4: Pairwise Pearson BEFORE residualization.
    # -----------------------------------------------------------------------
    tt_before = pairwise_upper_tri(target_mat)
    dd_before = pairwise_upper_tri(distractor_mat)
    td_before = cross_pairwise(target_mat, distractor_mat)

    # -----------------------------------------------------------------------
    # Step 5: Pairwise Pearson AFTER residualization.
    # -----------------------------------------------------------------------
    tt_after = pairwise_upper_tri(target_resid)
    dd_after = pairwise_upper_tri(distractor_resid)
    td_after = cross_pairwise(target_resid, distractor_resid)

    # -----------------------------------------------------------------------
    # Step 6: Save summary stats (before and after).
    # -----------------------------------------------------------------------
    before_rows = [
        condition_summary(tt_before, "target-target"),
        condition_summary(dd_before, "distractor-distractor"),
        condition_summary(td_before, "target-distractor"),
    ]
    after_rows = [
        condition_summary(tt_after, "target-target"),
        condition_summary(dd_after, "distractor-distractor"),
        condition_summary(td_after, "target-distractor"),
    ]

    pd.DataFrame(before_rows).to_csv(output_path / "ch9_pairwise_before.csv", index = False)
    pd.DataFrame(after_rows).to_csv(output_path  / "ch9_pairwise_after.csv",  index = False)

    # -----------------------------------------------------------------------
    # Step 7: Save raw residual pairwise values.
    # -----------------------------------------------------------------------
    pd.DataFrame({"r": tt_after}).to_csv(output_path / "ch9_tt_residual_raw.csv", index = False)
    pd.DataFrame({"r": dd_after}).to_csv(output_path / "ch9_dd_residual_raw.csv", index = False)
    pd.DataFrame({"r": td_after}).to_csv(output_path / "ch9_td_residual_raw.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 8: Before-vs-after comparison (Welch t-test, one-sample vs zero).
    # -----------------------------------------------------------------------
    comparison_rows = []
    for label, vals_before, vals_after in [
        ("target-target",         tt_before, tt_after),
        ("distractor-distractor", dd_before, dd_after),
        ("target-distractor",     td_before, td_after),
    ]:
        t_stat,  p_val   = ttest_ind(vals_before, vals_after, equal_var = False)
        t1_stat, t1_p    = ttest_1samp(vals_after, popmean = 0)
        wil_stat, wil_p  = wilcoxon(vals_after)

        comparison_rows.append({
            "condition":           label,
            "before_mean":         float(np.mean(vals_before)),
            "after_mean":          float(np.mean(vals_after)),
            "delta":               float(np.mean(vals_after) - np.mean(vals_before)),
            "welch_t_ba":          float(t_stat),
            "welch_p_ba":          float(p_val),
            "after_t_vs_zero":     float(t1_stat),
            "after_p_vs_zero":     float(t1_p),
            "after_wilcoxon_stat": float(wil_stat),
            "after_wilcoxon_p":    float(wil_p),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_path / "ch9_cohesion_comparison.csv", index = False)

    # -----------------------------------------------------------------------
    # Print summary to terminal.
    # -----------------------------------------------------------------------
    for row in comparison_rows:
        cond = row["condition"]
        print(
            f"{cond:>24s}  "
            f"before={row['before_mean']:+.4f}  "
            f"after={row['after_mean']:+.4f}  "
            f"delta={row['delta']:+.4f}  "
            f"welch_p={fmt_p(row['welch_p_ba'])}  "
            f"after_vs_0 p={fmt_p(row['after_p_vs_zero'])}",
            flush = True,
        )

    # -----------------------------------------------------------------------
    # Plot 1: Violin — projection coefficients by label.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6, 5))

    parts = ax.violinplot(
        [t_proj, d_proj],
        positions   = [1, 2],
        showmedians = True,
        showextrema = True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Targets", "Distractors"])
    ax.set_ylabel("OLS Projection Coefficient onto Wall Template")
    ax.set_title("Template Projection Coefficients by Label (Chapter 9)")
    ax.axhline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.grid(axis = "y", alpha = 0.3)

    # Annotate with mean ± std.
    for pos, vals in [(1, t_proj), (2, d_proj)]:
        ax.text(
            pos, ax.get_ylim()[0] + 0.01,
            f"M={np.mean(vals):.3f}\nSD={np.std(vals):.3f}",
            ha       = "center",
            va       = "bottom",
            fontsize = 8,
        )

    fig.tight_layout()
    fig.savefig(output_path / "ch9_projection_coefficients.png", dpi = 200)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: Grouped barplot — before vs after residualization per condition.
    # -----------------------------------------------------------------------
    conditions   = ["target-target", "distractor-distractor", "target-distractor"]
    short_labels = ["t-t", "d-d", "t-d"]
    before_means = [np.mean(tt_before), np.mean(dd_before), np.mean(td_before)]
    before_ses   = [np.std(tt_before, ddof=1)/np.sqrt(len(tt_before)),
                    np.std(dd_before, ddof=1)/np.sqrt(len(dd_before)),
                    np.std(td_before, ddof=1)/np.sqrt(len(td_before))]
    after_means  = [np.mean(tt_after),  np.mean(dd_after),  np.mean(td_after)]
    after_ses    = [np.std(tt_after,  ddof=1)/np.sqrt(len(tt_after)),
                    np.std(dd_after,  ddof=1)/np.sqrt(len(dd_after)),
                    np.std(td_after,  ddof=1)/np.sqrt(len(td_after))]

    x     = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize = (8, 5))
    bars_before = ax.bar(
        x - width / 2, before_means, width,
        yerr    = before_ses,
        capsize = 5,
        label   = "Before residualization",
        color   = "steelblue",
        alpha   = 0.85,
    )
    bars_after = ax.bar(
        x + width / 2, after_means, width,
        yerr    = after_ses,
        capsize = 5,
        label   = "After residualization",
        color   = "tomato",
        alpha   = 0.85,
    )

    # Annotate delta and significance above each pair.
    for i, row in enumerate(comparison_rows):
        y_top = max(before_means[i], after_means[i]) + max(before_ses[i], after_ses[i]) + 0.003
        ax.text(
            x[i], y_top,
            f"d={row['delta']:+.3f}  p={fmt_p(row['welch_p_ba'])}",
            ha       = "center",
            va       = "bottom",
            fontsize = 7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Mean Pairwise Pearson r")
    ax.set_title("Pairwise Cohesion Before and After Template Residualization (Chapter 9)")
    ax.axhline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.legend(fontsize = 9)
    ax.grid(axis = "y", alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9_before_after_cohesion.png", dpi = 200)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: Per-target scatter — projection coefficient vs residual cohesion.
    #   Asks: does template alignment predict how cohesive a target's residual is?
    # -----------------------------------------------------------------------
    # Residual mean_r for each target = mean Pearson r to all other targets
    # using the residualized waveforms.
    n_t               = target_resid.shape[0]
    full_resid_corr   = np.corrcoef(target_resid)
    diag_mask         = np.eye(n_t, dtype = bool)
    masked_corr       = np.where(diag_mask, np.nan, full_resid_corr)
    resid_mean_r      = np.nanmean(masked_corr, axis = 1)

    # Scatter.
    from scipy.stats import linregress
    slope, intercept, r_val, p_val, _ = linregress(t_proj, resid_mean_r)
    x_line = np.linspace(t_proj.min(), t_proj.max(), 300)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize = (7, 5))
    ax.scatter(t_proj, resid_mean_r, s = 20, alpha = 0.75, zorder = 3)
    ax.plot(
        x_line, y_line,
        color     = "black",
        linewidth = 1.5,
        label     = f"OLS  r = {r_val:.3f},  p = {fmt_p(p_val)}",
    )
    ax.axhline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.set_xlabel("OLS Projection Coefficient onto Wall Template")
    ax.set_ylabel("Mean Residual Pearson r to Other Targets")
    ax.set_title("Template Alignment vs Residual Cohesion (Chapter 9)")
    ax.legend(fontsize = 9)
    ax.grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9_proj_vs_residual_cohesion.png", dpi = 200)
    plt.close(fig)

    # Save per-target residual stats.
    per_target_df = targets_df[["stimulus_number"]].copy()
    per_target_df = per_target_df.reset_index(drop = True)
    per_target_df["projection_coeff"]  = t_proj
    per_target_df["resid_mean_r"]      = resid_mean_r
    per_target_df.to_csv(output_path / "ch9_per_target_residual_stats.csv", index = False)

    # -----------------------------------------------------------------------
    # Step 9: Decompose pairwise Pearson r into template vs residual contribution.
    #
    #   For centered s_i = alpha_i * t + e_i with e_i orthogonal to t:
    #     cov(s_i, s_j)     = alpha_i * alpha_j * var(t)  +  cov(e_i, e_j)
    #     Pearson(s_i, s_j) = [alpha_i*alpha_j*var(t) + cov(e_i, e_j)] / (sd(s_i)*sd(s_j))
    #                       = template_contrib + residual_contrib
    #   No z-normalization assumed; identity holds for the raw on-disk waveforms.
    # -----------------------------------------------------------------------
    n_t = target_mat.shape[0]
    n_d = distractor_mat.shape[0]

    var_template = float(np.var(template, ddof = 1))     # var of centered wall

    # Stimulus and residual standard deviations (ddof=1 to match np.corrcoef).
    sd_t   = np.std(target_mat,       axis = 1, ddof = 1)
    sd_d   = np.std(distractor_mat,   axis = 1, ddof = 1)
    sd_e_t = np.std(target_resid,     axis = 1, ddof = 1)
    sd_e_d = np.std(distractor_resid, axis = 1, ddof = 1)

    # Full correlation matrices on original (centered) stimuli.
    tt_corr_full = np.corrcoef(target_mat)
    dd_corr_full = np.corrcoef(distractor_mat)
    td_stacked   = np.vstack([target_mat, distractor_mat])
    td_corr_full = np.corrcoef(td_stacked)[:n_t, n_t:]

    # Residual correlation matrices.
    tt_resid_corr = np.corrcoef(target_resid)
    dd_resid_corr = np.corrcoef(distractor_resid)
    td_resid_stacked = np.vstack([target_resid, distractor_resid])
    td_resid_corr    = np.corrcoef(td_resid_stacked)[:n_t, n_t:]

    tt_idx = np.triu_indices(n_t, k = 1)
    dd_idx = np.triu_indices(n_d, k = 1)

    def template_contrib_matrix(alpha_a, alpha_b, sd_a, sd_b):
        """alpha_i * alpha_j * var(t) / (sd_a_i * sd_b_j) for all pairs (i, j)."""
        num = np.outer(alpha_a, alpha_b) * var_template
        den = np.outer(sd_a, sd_b)
        return num / den

    def residual_contrib_matrix(resid_corr, sd_e_a, sd_e_b, sd_a, sd_b):
        """cov(e_i, e_j) / (sd_a_i * sd_b_j) for all pairs (i, j),
        derived from Pearson(e_i, e_j) and the residual sds."""
        cov_resid = resid_corr * np.outer(sd_e_a, sd_e_b)
        return cov_resid / np.outer(sd_a, sd_b)

    # tt decomposition.
    tt_total_r = tt_corr_full[tt_idx]
    tt_tmpl    = template_contrib_matrix(t_proj, t_proj, sd_t, sd_t)[tt_idx]
    tt_resid_c = residual_contrib_matrix(tt_resid_corr, sd_e_t, sd_e_t, sd_t, sd_t)[tt_idx]

    # dd decomposition.
    dd_total_r = dd_corr_full[dd_idx]
    dd_tmpl    = template_contrib_matrix(d_proj, d_proj, sd_d, sd_d)[dd_idx]
    dd_resid_c = residual_contrib_matrix(dd_resid_corr, sd_e_d, sd_e_d, sd_d, sd_d)[dd_idx]

    # td decomposition (all n_t * n_d pairs).
    td_total_r = td_corr_full.ravel()
    td_tmpl    = template_contrib_matrix(t_proj, d_proj, sd_t, sd_d).ravel()
    td_resid_c = residual_contrib_matrix(td_resid_corr, sd_e_t, sd_e_d, sd_t, sd_d).ravel()

    # Sanity check: each pair's total r must equal template_contrib + residual_contrib.
    max_err = max(
        np.abs(tt_total_r - (tt_tmpl + tt_resid_c)).max(),
        np.abs(dd_total_r - (dd_tmpl + dd_resid_c)).max(),
        np.abs(td_total_r - (td_tmpl + td_resid_c)).max(),
    )
    print(f"Decomposition max abs error (should be ~0): {max_err:.2e}", flush = True)
    assert max_err < 1e-6, f"Decomposition identity failed: max err {max_err}"

    # Summary stats per condition.
    decomp_rows = []
    for label, total, tmpl, resid in [
        ("target-target",         tt_total_r, tt_tmpl, tt_resid_c),
        ("distractor-distractor", dd_total_r, dd_tmpl, dd_resid_c),
        ("target-distractor",     td_total_r, td_tmpl, td_resid_c),
    ]:
        decomp_rows.append({
            "condition":         label,
            "mean_total_r":      float(np.mean(total)),
            "mean_tmpl_contrib": float(np.mean(tmpl)),
            "mean_resid_contrib":float(np.mean(resid)),
            "frac_tmpl":         float(np.mean(tmpl) / np.mean(total)) if np.mean(total) != 0 else np.nan,
        })
        print(
            f"{label:>24s}  total={np.mean(total):+.5f}  "
            f"template={np.mean(tmpl):+.5f}  "
            f"residual={np.mean(resid):+.5f}",
            flush = True,
        )

    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(output_path / "ch9_pearson_decomposition_summary.csv", index = False)

    # Per-pair raw tables.
    pd.DataFrame({
        "r_total":          tt_total_r,
        "template_contrib": tt_tmpl,
        "residual_contrib": tt_resid_c,
    }).to_csv(output_path / "ch9_tt_decomposition_raw.csv", index = False)

    pd.DataFrame({
        "r_total":          dd_total_r,
        "template_contrib": dd_tmpl,
        "residual_contrib": dd_resid_c,
    }).to_csv(output_path / "ch9_dd_decomposition_raw.csv", index = False)

    pd.DataFrame({
        "r_total":          td_total_r,
        "template_contrib": td_tmpl,
        "residual_contrib": td_resid_c,
    }).to_csv(output_path / "ch9_td_decomposition_raw.csv", index = False)

    # -----------------------------------------------------------------------
    # Plot 4: Stacked bar — mean template vs residual contribution per condition.
    # -----------------------------------------------------------------------
    short_labels = ["t-t", "d-d", "t-d"]
    tmpl_means   = [r["mean_tmpl_contrib"]  for r in decomp_rows]
    resid_means  = [r["mean_resid_contrib"] for r in decomp_rows]
    total_means  = [r["mean_total_r"]       for r in decomp_rows]

    x     = np.arange(len(short_labels))
    width = 0.5

    fig, ax = plt.subplots(figsize = (7, 5))
    ax.bar(x, tmpl_means,  width, label = "Template contribution", color = "goldenrod", alpha = 0.9)
    ax.bar(x, resid_means, width, bottom = tmpl_means, label = "Residual contribution", color = "steelblue", alpha = 0.9)

    # Total r tick marks.
    for i, tot in enumerate(total_means):
        ax.hlines(tot, i - width / 2, i + width / 2, colors = "black", linewidths = 1.5,
                  label = "Total r" if i == 0 else None)
        ax.text(i + width / 2 + 0.03, tot, f"{tot:+.4f}", va = "center", fontsize = 8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Mean Pairwise Pearson r")
    ax.set_title("Decomposition of Pairwise r: Template vs Residual (Chapter 9)")
    ax.axhline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.legend(fontsize = 9)
    ax.grid(axis = "y", alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_path / "ch9_pearson_decomposition_bar.png", dpi = 200)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 5: Distribution of template contributions — tt vs dd vs td.
    #   Answers: are tt template contributions distinguishably larger?
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize = (12, 4), sharey = False)
    for ax, label, tmpl_vals in [
        (axes[0], "target-target",         tt_tmpl),
        (axes[1], "distractor-distractor", dd_tmpl),
        (axes[2], "target-distractor",     td_tmpl),
    ]:
        ax.hist(tmpl_vals, bins = 50, color = "goldenrod", alpha = 0.85, edgecolor = "none")
        ax.axvline(np.mean(tmpl_vals), color = "black", linewidth = 1.5, linestyle = "--",
                   label = f"mean={np.mean(tmpl_vals):.5f}")
        ax.set_title(label)
        ax.set_xlabel("Template Contribution (alpha_i * alpha_j)")
        ax.set_ylabel("Count")
        ax.legend(fontsize = 8)
        ax.grid(alpha = 0.3)

    fig.suptitle("Distribution of Template-Mediated Pairwise r (Chapter 9)", y = 1.01)
    fig.tight_layout()
    fig.savefig(output_path / "ch9_template_contribution_distributions.png", dpi = 200, bbox_inches = "tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 6: Scatter — template contrib vs total r for tt pairs.
    #   Color = residual contribution.  Shows how much total r is driven by template.
    # -----------------------------------------------------------------------
    from scipy.stats import linregress as _lr
    sl, ic, rv, pv, _ = _lr(tt_tmpl, tt_total_r)
    xl = np.linspace(tt_tmpl.min(), tt_tmpl.max(), 300)

    fig, ax = plt.subplots(figsize = (7, 5))
    sc = ax.scatter(tt_tmpl, tt_total_r, c = tt_resid_c, cmap = "coolwarm",
                    s = 15, alpha = 0.7, zorder = 3)
    ax.plot(xl, sl * xl + ic, color = "black", linewidth = 1.5,
            label = f"OLS  r = {rv:.3f},  p = {fmt_p(pv)}")
    ax.axhline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.axvline(0, color = "gray", linewidth = 0.8, linestyle = "--")
    ax.set_xlabel("Template Contribution (alpha_i * alpha_j)")
    ax.set_ylabel("Total Pairwise Pearson r")
    ax.set_title("Template Contribution vs Total r — Target Pairs (Chapter 9)")
    ax.legend(fontsize = 9)
    ax.grid(alpha = 0.3)
    cbar = fig.colorbar(sc, ax = ax, fraction = 0.046, pad = 0.04)
    cbar.set_label("Residual Contribution")

    fig.subplots_adjust(left = 0.12, right = 0.87, bottom = 0.10, top = 0.93)
    fig.savefig(output_path / "ch9_tt_template_vs_total_r.png", dpi = 200)
    plt.close(fig)

    print("Chapter 9 done.  Outputs saved to:", output_path, flush = True)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Chapter 9: projection onto the template and residual structure."""
    cur_path    = Path(__file__).parent.resolve()
    output_path = cur_path / "outputs" / "pearson" / "ch9-projection-residual"

    targets_df, distractors_df, wall_wav = load_stimuli(cur_path)
    chapter_9(targets_df, distractors_df, wall_wav, output_path)


if __name__ == "__main__":
    main()
