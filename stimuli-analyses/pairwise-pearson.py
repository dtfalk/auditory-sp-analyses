from pathlib import Path
import numpy as np
import logging
from scipy.io import wavfile
from scipy.stats import pearsonr 
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon
import pandas as pd
import matplotlib.pyplot as plt


def fmt_p(p_val):
    """Format a p-value as decimal unless it is smaller than 1e-4.

    Args:
        p_val: Float p-value.

    Returns:
        Formatted string.
    """
    if np.isnan(p_val):
        return "     nan"
    if p_val < 1e-4:
        return f"{p_val:.3e}"
    return f"{p_val:.4f}"


# ============================================================================
# General Helpers
# ============================================================================
def get_stimuli(cur_path):
    """Loads the stimuli"""

    # Audio stimuli paths
    #   - "fs" = "Full Sentence"
    #   - "is" = "Imagined Sentence"
    stimuli_path = cur_path / ".." / "raw_data" / "audio_stimuli"
    fs_targs_paths = stimuli_path / "full_sentence" / "targets"
    fs_dists_paths = stimuli_path / "full_sentence" / "distractors"
    is_targs_paths = stimuli_path / "imagined_sentence" / "targets"
    is_dists_paths = stimuli_path / "imagined_sentence" / "distractors"

    # Extract data and verify stimuli properties
    fs_targets     = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in fs_targs_paths.glob("*.wav")]]
    fs_distractors = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in fs_dists_paths.glob("*.wav")]]
    is_targets     = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in is_targs_paths.glob("*.wav")]]
    is_distractors = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in is_dists_paths.glob("*.wav")]]   
    

    # Load the stimuli & extract sampling rate and the raw data as we go
    #   Place it all in a dict for ez access
    stimuli = {}
    stimuli["fs_targets"]     = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in fs_targs_paths.glob("*.wav")]]
    stimuli["fs_distractors"] = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in fs_dists_paths.glob("*.wav")]]
    stimuli["is_targets"]     = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in is_targs_paths.glob("*.wav")]]
    stimuli["is_distractors"] = [(fpath.stem, rate, data.astype(np.float32)) for fpath, (rate, data) in [(fpath, wavfile.read(fpath)) for fpath in is_dists_paths.glob("*.wav")]]
    stimuli["full_sentence"]  = fs_targets + fs_distractors
    stimuli["imag_sentence"]  = is_targets + is_distractors
    stimuli["targets"]        = fs_targets + is_targets
    stimuli["distractors"]    = fs_distractors + is_distractors

    return stimuli


def verify_stimuli_properties(stimuli):
    """Confirms the stimuli lists are equal length, each stimulus has same sampling rate, number of samples, and channel width""" 
    
    # Confirm equal nums of stimuli and samp rates
    keys = ["fs_targets", "fs_distractors", "is_targets", "is_distractors"]
    for key in keys:
        
        cur_stim_list = stimuli[key]

        # Check that all stimuli lists have the same length as the first set in the list
        #   Kinda hacky but we have diff lengths so can't pick one specific length to check against
        assert len(cur_stim_list) == 75, f"Unexpected Number of Stimuli"

        # Create bool lists to handle samp rate and num samples and confirm uniformity
        sampRate_bools  = [samp_rate == 8000 for _, samp_rate, _ in cur_stim_list]
        dataLen_bools   = [len(data) == 2960 for _,  _, data in cur_stim_list]
        chanWidth_bools = [data.ndim == 1 for _, _, data in cur_stim_list]
        assert all(sampRate_bools) and all(dataLen_bools) and all(chanWidth_bools), "Invalid Sampling Rate (Expected: 8000), Stimulus Length (Expected: 2960), or Channel Width (Expected: 1)"

# ============================================================================
# ============================================================================


# ============================================================================
# Pairwise Pearson 
# ============================================================================
# Pairwise pearson scores
def get_pairwise_pearson_scores(stimuli, all_data_results_path, summary_data_results_path):
    
    pairwise_results = {}

    # Normalize + stack
    targ_matrix = np.vstack([
        (data - np.mean(data)) / np.std(data)
        for _, _, data in stimuli["targets"]
    ])

    dist_matrix = np.vstack([
        (data - np.mean(data)) / np.std(data)
        for _, _, data in stimuli["distractors"]
    ])

   # Correlation matrices
    tt = np.corrcoef(targ_matrix)
    dd = np.corrcoef(dist_matrix)
    td = np.corrcoef(targ_matrix, dist_matrix)

    # Extract upper triangle (no diagonal)
    tt_vals = tt[np.triu_indices_from(tt, k = 1)]
    dd_vals = dd[np.triu_indices_from(dd, k = 1)]

    # For cross condition, take top-right block
    n_t = targ_matrix.shape[0]
    n_d = dist_matrix.shape[0]
    td_vals = td[:n_t, n_t:n_t+n_d].flatten()

    # Save raw data
    pd.DataFrame(tt_vals, columns = ["pearson_r"]).to_csv(
        all_data_results_path / "target_target_raw.csv", index = False
    )
    pd.DataFrame(dd_vals, columns = ["pearson_r"]).to_csv(
        all_data_results_path / "distractor_distractor_raw.csv", index = False
    )
    pd.DataFrame(td_vals, columns = ["pearson_r"]).to_csv(
        all_data_results_path / "target_distractor_raw.csv", index = False
    )

    # Compute stats
    tt_mean = np.mean(tt_vals)
    tt_std  = np.std(tt_vals)
    tt_se   = tt_std / np.sqrt(len(tt_vals))
    tt_min  = np.min(tt_vals)
    tt_max  = np.max(tt_vals)

    dd_mean = np.mean(dd_vals)
    dd_std  = np.std(dd_vals)
    dd_se   = dd_std / np.sqrt(len(dd_vals))
    dd_min  = np.min(dd_vals)
    dd_max  = np.max(dd_vals)

    td_mean = np.mean(td_vals)
    td_std  = np.std(td_vals)
    td_se   = td_std / np.sqrt(len(td_vals))
    td_min  = np.min(td_vals)
    td_max  = np.max(td_vals)

    # Welch's t-tests for quick differences in means
    ttest_results = []
    condition_pairs = [
        ("target-target", tt_vals, "distractor-distractor", dd_vals),
        ("target-target", tt_vals, "target-distractor", td_vals),
        ("distractor-distractor", dd_vals, "target-distractor", td_vals),
    ]

    for cond_a, vals_a, cond_b, vals_b in condition_pairs:
        t_stat, p_val = ttest_ind(vals_a, vals_b, equal_var = False)
        mean_diff = np.mean(vals_a) - np.mean(vals_b)

        ttest_results.append([
            cond_a,
            cond_b,
            mean_diff,
            t_stat,
            p_val,
        ])

    ttest_df = pd.DataFrame(
        ttest_results,
        columns = ["condition_a", "condition_b", "mean_diff_a_minus_b", "t_stat", "p_value"],
    )

    # Save summary
    summary_df = pd.DataFrame([
        ["target-target", tt_mean, tt_std, tt_se, tt_min, tt_max, len(tt_vals)],
        ["distractor-distractor", dd_mean, dd_std, dd_se, dd_min, dd_max, len(dd_vals)],
        ["target-distractor", td_mean, td_std, td_se, td_min, td_max, len(td_vals)],
    ], columns = ["condition", "mean", "std", "se", "min", "max", "n"])

    all_data_df = {
        "target-target": pd.DataFrame(tt_vals, columns = ["pearson_r"]),
        "distractor-distractor": pd.DataFrame(dd_vals, columns = ["pearson_r"]),
        "target-distractor": pd.DataFrame(td_vals, columns = ["pearson_r"]),
    }

    pairwise_results = {
        "all-data": all_data_df, 
        "summary-data": summary_df
    }

    # Save summary data
    summary_df.to_csv(summary_data_results_path / "summary_stats.csv", index = False)
    ttest_df.to_csv(summary_data_results_path / "welch_t_tests.csv", index = False)

    # Save detailed distribution summary stats
    dist_summary_df = pd.DataFrame([
        ["target-target", np.mean(tt_vals), np.median(tt_vals), np.var(tt_vals), len(tt_vals)],
        ["distractor-distractor", np.mean(dd_vals), np.median(dd_vals), np.var(dd_vals), len(dd_vals)],
        ["target-distractor", np.mean(td_vals), np.median(td_vals), np.var(td_vals), len(td_vals)],
    ], columns = ["condition", "mean", "median", "variance", "n"])
    dist_summary_df.to_csv(summary_data_results_path / "distribution_summary_stats.csv", index = False)

    # Save effect sizes for pairwise condition comparisons
    effect_size_rows = []
    for cond_a, vals_a, cond_b, vals_b in condition_pairs:
        n_a = len(vals_a)
        n_b = len(vals_b)

        var_a = np.var(vals_a, ddof = 1)
        var_b = np.var(vals_b, ddof = 1)

        pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        cohens_d = (np.mean(vals_a) - np.mean(vals_b)) / pooled_sd
        hedges_g = cohens_d * (1 - (3 / (4 * (n_a + n_b) - 9)))

        effect_size_rows.append([
            cond_a,
            cond_b,
            np.mean(vals_a) - np.mean(vals_b),
            cohens_d,
            hedges_g,
        ])

    effect_sizes_df = pd.DataFrame(
        effect_size_rows,
        columns = ["condition_a", "condition_b", "mean_diff_a_minus_b", "cohens_d", "hedges_g"],
    )
    effect_sizes_df.to_csv(summary_data_results_path / "effect_sizes.csv", index = False)

    # Test each distribution against zero.
    one_sample_rows = []
    for condition, vals in [
        ("target-target", tt_vals),
        ("distractor-distractor", dd_vals),
        ("target-distractor", td_vals),
    ]:
        t_stat, t_p_val = ttest_1samp(vals, popmean = 0.0)

        # Wilcoxon can fail when differences are all zeros or identical.
        try:
            w_stat, w_p_val = wilcoxon(vals)
        except ValueError:
            w_stat, w_p_val = np.nan, np.nan

        one_sample_rows.append([
            condition,
            np.mean(vals),
            t_stat,
            t_p_val,
            w_stat,
            w_p_val,
            len(vals),
        ])

    one_sample_df = pd.DataFrame(
        one_sample_rows,
        columns = [
            "condition",
            "mean",
            "t_stat_vs_zero",
            "t_p_value_vs_zero",
            "wilcoxon_stat_vs_zero",
            "wilcoxon_p_value_vs_zero",
            "n",
        ],
    )
    one_sample_df.to_csv(summary_data_results_path / "one_sample_tests_vs_zero.csv", index = False)

    # Build a full correlation matrix with targets first, distractors second for heatmap plotting
    full_group_matrix = np.vstack([targ_matrix, dist_matrix])
    full_group_corr   = np.corrcoef(full_group_matrix)
    group_labels      = ["target"] * targ_matrix.shape[0] + ["distractor"] * dist_matrix.shape[0]
    pd.DataFrame(full_group_corr).to_csv(all_data_results_path / "correlation_matrix_group_ordered.csv", index = False)

    return pairwise_results

def main():

    # Current directory
    cur_path = Path(__file__).parent.resolve()
    
    # Output folders
    summary_data_results_path     = cur_path / "outputs" / "pearson" / "pariwise" / "summary"
    all_data_results_path         = cur_path / "outputs" / "pearson" / "pariwise" / "all_data"
    summary_data_results_path.mkdir(parents = True, exist_ok = True)
    all_data_results_path.mkdir(parents = True, exist_ok = True)
    
    # Get the stimuli 
    stimuli = get_stimuli(cur_path)

    # Verify all stimuli propertes are as expected
    verify_stimuli_properties(stimuli)
    
    # Get the results of the pairwise analyses
    pairwise_results = get_pairwise_pearson_scores(stimuli, all_data_results_path, summary_data_results_path)


    # Plot
    fig, (ax_bar, ax_text) = plt.subplots(
        nrows       = 2,
        ncols       = 1,
        figsize     = (12, 9),
        gridspec_kw = {"height_ratios": [4, 1]},
    )

    bars = ax_bar.bar(
        pairwise_results["summary-data"]["condition"],
        pairwise_results["summary-data"]["mean"],
        yerr    = pairwise_results["summary-data"]["se"],
        capsize = 5,
    )

    # Recompute Welch text for plot annotation
    tt_vals = pairwise_results["all-data"]["target-target"]["pearson_r"].to_numpy()
    dd_vals = pairwise_results["all-data"]["distractor-distractor"]["pearson_r"].to_numpy()
    td_vals = pairwise_results["all-data"]["target-distractor"]["pearson_r"].to_numpy()

    # Load detailed summary/effect-size outputs for display text
    dist_summary_df = pd.read_csv(summary_data_results_path / "distribution_summary_stats.csv")
    effect_sizes_df = pd.read_csv(summary_data_results_path / "effect_sizes.csv")
    one_sample_df = pd.read_csv(summary_data_results_path / "one_sample_tests_vs_zero.csv")

    welch_lines = [
        "comparison        mean_diff  t_stat    p_value",
    ]
    for cond_a, vals_a, cond_b, vals_b in [
        ("t-t", tt_vals, "d-d", dd_vals),
        ("t-t", tt_vals, "t-d", td_vals),
        ("d-d", dd_vals, "t-d", td_vals),
    ]:
        t_stat, p_val = ttest_ind(vals_a, vals_b, equal_var = False)
        mean_diff = np.mean(vals_a) - np.mean(vals_b)
        welch_lines.append(f"{cond_a} vs {cond_b:<7} {mean_diff:>9.4f}  {t_stat:>9.3f}  {fmt_p(p_val):>9}")

    # Build compact summary text block below the bar chart
    summary_lines = [
        "condition                 mean      min      max",
    ]
    for _, row in pairwise_results["summary-data"].iterrows():
        summary_lines.append(
            f"{row['condition']:<24} {row['mean']:>7.4f}  {row['min']:>7.4f}  {row['max']:>7.4f}",
        )

    dist_lines = [
        "condition                 mean    median   variance",
    ]
    for _, row in dist_summary_df.iterrows():
        dist_lines.append(
            f"{row['condition']:<24} {row['mean']:>7.4f}  {row['median']:>7.4f}  {row['variance']:>9.6f}",
        )

    effect_lines = [
        "comparison    cohens_d   hedges_g",
    ]
    effect_label_map = {
        "target-target": "t-t",
        "distractor-distractor": "d-d",
        "target-distractor": "t-d",
    }
    for _, row in effect_sizes_df.iterrows():
        cond_a = effect_label_map.get(row["condition_a"], row["condition_a"])
        cond_b = effect_label_map.get(row["condition_b"], row["condition_b"])
        comparison = f"{cond_a} vs {cond_b}"
        effect_lines.append(f"{comparison}    {row['cohens_d']:.4f}     {row['hedges_g']:.4f}")

    one_sample_lines = [
        "condition                 t_p_value   wilcoxon_p",
    ]
    for _, row in one_sample_df.iterrows():
        one_sample_lines.append(
            f"{row['condition']:<24} {fmt_p(row['t_p_value_vs_zero']):>10}  {fmt_p(row['wilcoxon_p_value_vs_zero']):>10}",
        )

    ax_text.axis("off")
    ax_text.text(
        0.53,
        5.20,
        "Summary Stats:\n"
        + "\n".join(summary_lines)
        + "\n\nDistribution Stats:\n"
        + "\n".join(dist_lines)
        + "\n\nEffect Sizes:\n"
        + "\n".join(effect_lines)
        + "\n\nTests vs Zero:\n"
        + "\n".join(one_sample_lines)
        + "\n\nWelch comparisons (difference of means)\n"
        + "\n".join(welch_lines),
        family   = "monospace",
        fontsize = 8,
        va       = "top",
        ha       = "left",
        transform = ax_text.transAxes,
    )

    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title("Pairwise Correlations By Stimulus Group (Mean +/- SE)")
    fig.subplots_adjust(hspace = 0.15)
    fig.savefig(summary_data_results_path / "pearson_barplot.png", dpi = 200)
    plt.close(fig)

    # Plot distribution histograms for each condition
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))

    conditions = [
        ("target-target", tt_vals),
        ("distractor-distractor", dd_vals),
        ("target-distractor", td_vals),
    ]

    for ax, (label, vals) in zip(axes, conditions):
        ax.hist(vals, bins = 20, edgecolor = "black", alpha = 0.7)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"{label}\n"
            f"n = {len(vals)}, mean = {np.mean(vals):.4f},\n"
            f"median = {np.median(vals):.4f}, var = {np.var(vals):.6f}",
        )
        ax.grid(axis = "y", alpha = 0.3)

    fig.suptitle("Distribution of Pairwise Correlations")
    fig.tight_layout()
    fig.savefig(all_data_results_path / "distributions_histograms.png", dpi = 200)
    plt.close(fig)

    # Plot correlation matrix heatmap ordered by group (targets then distractors)
    full_group_corr = pd.read_csv(all_data_results_path / "correlation_matrix_group_ordered.csv").to_numpy()
    n_targets = len(stimuli["targets"])

    # Normalize color scale to the largest non-diagonal, non-1 magnitude.
    diag_mask = np.eye(full_group_corr.shape[0], dtype = bool)
    off_diag_vals = full_group_corr[~diag_mask]
    valid_scale_vals = off_diag_vals[~np.isclose(np.abs(off_diag_vals), 1.0)]
    max_abs_off_diag = np.max(np.abs(valid_scale_vals)) if valid_scale_vals.size > 0 else 1.0

    # Make diagonal black so off-diagonal structure is easier to see.
    heatmap_matrix = full_group_corr.copy()
    np.fill_diagonal(heatmap_matrix, np.nan)
    heatmap_cmap = plt.cm.coolwarm.copy()
    heatmap_cmap.set_bad(color = "black")

    fig, ax = plt.subplots(figsize = (9, 8))
    img = ax.imshow(
        heatmap_matrix,
        cmap = heatmap_cmap,
        vmin = -max_abs_off_diag,
        vmax = max_abs_off_diag,
        origin = "lower",
    )

    ax.axhline(n_targets - 0.5, color = "black", linewidth = 1.0)
    ax.axvline(n_targets - 0.5, color = "black", linewidth = 1.0)

    ax.set_title("Correlation Matrix (Targets Then Distractors)")
    ax.set_xlabel("Stimulus Index")
    ax.set_ylabel("Stimulus Index")

    cbar = fig.colorbar(img, ax = ax, fraction = 0.046, pad = 0.04)
    cbar.set_label("Pearson r (normalized to max |off-diagonal|)")

    fig.subplots_adjust(left = 0.13, right = 0.93, bottom = 0.10, top = 0.93)
    fig.savefig(all_data_results_path / "correlation_matrix_group_ordered_heatmap.png", dpi = 200)
    plt.close(fig)
        
    
    
        
 
if __name__ == "__main__":
    main()