from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.linear_model import BayesianRidge
import pandas as pd
import matplotlib.pyplot as plt


def hpdi(samples, prob = 0.94):
    """Compute the Highest Posterior Density Interval for a 1-D sample array.

    Args:
        samples: 1-D numpy array of posterior draws.
        prob:    Probability mass to enclose (default 0.94).

    Returns:
        Tuple (low, high) bounding the shortest interval containing prob mass.
    """
    sorted_s       = np.sort(samples)
    n              = len(sorted_s)
    interval_width = int(np.floor(prob * n))
    widths         = sorted_s[interval_width:] - sorted_s[: n - interval_width]
    min_idx        = np.argmin(widths)
    return sorted_s[min_idx], sorted_s[min_idx + interval_width]


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


# ============================================================================
# Constants
# ============================================================================
STEP_SIZE = 10
SUBSET_SIZES = [STEP_SIZE * (n + 1) for n in range(150 // STEP_SIZE)]

# ============================================================================
# Data Loading
# ============================================================================

def load_ranked_targets(cur_path):
    """Load target wavs joined with their template Pearson rank scores.

    Args:
        cur_path: Path to the script directory.

    Returns:
        DataFrame with columns [stimulus_number, r_score, waveform] sorted
        by r_score descending (highest template alignment first).
    """
    stimuli_path  = cur_path / ".." / "raw_data" / "audio_stimuli"
    rank_csv_path = stimuli_path / "correlation_csvs" / "high_correlation_stimuli.csv"
    fs_targs_path = stimuli_path / "full_sentence" / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"

    rank_df = pd.read_csv(rank_csv_path)

    wav_records = []
    for wav_path in list(fs_targs_path.glob("*.wav")) + list(is_targs_path.glob("*.wav")):
        _, raw       = wavfile.read(wav_path)
        data         = raw.astype(np.float32)
        normalized   = (data - np.mean(data)) / np.std(data)
        wav_records.append({"stimulus_number": int(wav_path.stem), "waveform": normalized})

    wav_df = pd.DataFrame(wav_records)
    merged = rank_df.merge(wav_df, on = "stimulus_number", how = "inner")
    merged = merged.sort_values("r_score", ascending = False).reset_index(drop = True)

    return merged


# ============================================================================
# Chapter 6
# ============================================================================

def chapter_6_cohesion_by_rank(ranked_df, output_path):
    """Compute and plot within-group pairwise cohesion for increasing top-N subsets.

    Args:
        ranked_df:   DataFrame sorted by r_score descending, with waveform and
                     mean_r_to_others columns already attached.
        output_path: Path to save outputs.

    Returns:
        DataFrame of subset cohesion stats.
    """
    # Per-subset cohesion stats.
    waveforms     = np.vstack(ranked_df["waveform"].to_list())
    cohesion_rows = []

    for subset_size in SUBSET_SIZES:
        top_n = waveforms[:subset_size]
        corr  = np.corrcoef(top_n)
        vals  = corr[np.triu_indices_from(corr, k = 1)]

        cohesion_rows.append({
            "subset_size": subset_size,
            "mean_r":      np.mean(vals),
            "se_r":        np.std(vals, ddof = 1) / np.sqrt(len(vals)),
            "std_r":       np.std(vals, ddof = 1),
            "n_pairs":     len(vals),
        })

    cohesion_df = pd.DataFrame(cohesion_rows)
    cohesion_df.to_csv(output_path / "ch6_subset_cohesion.csv", index = False)
    ranked_df[["stimulus_number", "r_score", "mean_r_to_others"]].to_csv(
        output_path / "ch6_per_target_mean_r.csv", index = False
    )

    # Plot 1: cohesion curve (subset size vs mean pairwise r).
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.errorbar(
        cohesion_df["subset_size"],
        cohesion_df["mean_r"],
        yerr    = cohesion_df["se_r"],
        marker  = "o",
        capsize = 5,
    )
    ax.set_xlabel("Top-N Subset Size")
    ax.set_ylabel("Mean Pairwise Pearson r")
    ax.set_title(f"Target Cohesion by Rank [Step Size = {STEP_SIZE}]")
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ch6_cohesion_curve.png", dpi = 200)
    plt.close(fig)

    # Plot 2: rank vs mean similarity to full target set, with regression line.
    ranks = (ranked_df.index + 1).to_numpy()
    y_sim = ranked_df["mean_r_to_others"].to_numpy()

    slope, intercept, r_val, p_val, _ = linregress(ranks, y_sim)
    x_line = np.linspace(ranks.min(), ranks.max(), 300)
    y_line = slope * x_line + intercept

    # Bayesian ridge: near-flat Gamma hyperpriors so the data drives the fit.
    #   alpha_1/alpha_2 = noise precision prior (Gamma shape/rate)
    #   lambda_1/lambda_2 = weight precision prior (Gamma shape/rate)
    #   1e-10 makes both Gammas almost uniform; posterior ~= pure likelihood.
    bayes_model = BayesianRidge(
        alpha_1  = 1e-10,
        alpha_2  = 1e-10,
        lambda_1 = 1e-10,
        lambda_2 = 1e-10,
    )
    bayes_model.fit(ranks.reshape(-1, 1), y_sim)
    y_bayes = bayes_model.predict(x_line.reshape(-1, 1))

    # Sample from posterior over weights N(coef_, sigma_) to get HPDI band.
    rng            = np.random.default_rng(seed = 42)
    weight_samples = rng.multivariate_normal(bayes_model.coef_, bayes_model.sigma_, size = 2000)
    pred_samples   = weight_samples @ x_line.reshape(1, -1) + bayes_model.intercept_
    hpdi_low       = np.array([hpdi(pred_samples[:, i])[0] for i in range(len(x_line))])
    hpdi_high      = np.array([hpdi(pred_samples[:, i])[1] for i in range(len(x_line))])

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.scatter(ranks, y_sim, s = 15, alpha = 0.7, zorder = 3)
    ax.plot(
        x_line, y_line,
        color     = "black",
        linewidth = 1.5,
        label     = f"OLS  r = {r_val:.3f},  p = {fmt_p(p_val)}",
    )
    ax.plot(
        x_line, y_bayes,
        color     = "green",
        linewidth = 1.5,
        label     = "Bayesian ridge (estimated priors)",
    )
    ax.fill_between(
        x_line,
        hpdi_low,
        hpdi_high,
        color = "green",
        alpha = 0.15,
        label = "94% HPDI",
    )
    ax.legend(fontsize = 8)
    ax.set_xlabel("Rank (1 = highest template correlation)")
    ax.set_ylabel("Mean Pearson r to Other Targets")
    ax.set_title("Rank vs Mean Similarity to Target Set (Chapter 6)")
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ch6_rank_vs_mean_similarity.png", dpi = 200)
    plt.close(fig)

    return cohesion_df


# ============================================================================
# Chapter 7
# ============================================================================

def chapter_7_template_vs_cohesion(ranked_df, output_path):
    """Scatterplot and regression of template Pearson vs mean pairwise cohesion.

    Args:
        ranked_df:   DataFrame with r_score and mean_r_to_others columns.
        output_path: Path to save outputs.

    Returns:
        None
    """
    x = ranked_df["r_score"].to_numpy()
    y = ranked_df["mean_r_to_others"].to_numpy()

    slope, intercept, r_val, p_val, se = linregress(x, y)
    y_pred    = slope * x + intercept
    residuals = y - y_pred

    # Save regression summary and residuals.
    pd.DataFrame([{
        "slope":     slope,
        "intercept": intercept,
        "r_value":   r_val,
        "r_squared": r_val ** 2,
        "p_value":   p_val,
        "std_err":   se,
    }]).to_csv(output_path / "ch7_regression_summary.csv", index = False)

    ranked_df.assign(y_pred = y_pred, residual = residuals)[
        ["stimulus_number", "r_score", "mean_r_to_others", "y_pred", "residual"]
    ].to_csv(output_path / "ch7_residuals.csv", index = False)

    # Scatterplot with regression line, residuals as color.
    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = slope * x_line + intercept

    # Bayesian ridge: near-flat Gamma hyperpriors so the data drives the fit.
    bayes_model = BayesianRidge(
        alpha_1  = 1e-10,
        alpha_2  = 1e-10,
        lambda_1 = 1e-10,
        lambda_2 = 1e-10,
    )
    bayes_model.fit(x.reshape(-1, 1), y)
    y_bayes = bayes_model.predict(x_line.reshape(-1, 1))

    # Sample from posterior over weights N(coef_, sigma_) to get HPDI band.
    rng            = np.random.default_rng(seed = 42)
    weight_samples = rng.multivariate_normal(bayes_model.coef_, bayes_model.sigma_, size = 2000)
    pred_samples   = weight_samples @ x_line.reshape(1, -1) + bayes_model.intercept_
    hpdi_low       = np.array([hpdi(pred_samples[:, i])[0] for i in range(len(x_line))])
    hpdi_high      = np.array([hpdi(pred_samples[:, i])[1] for i in range(len(x_line))])

    fig, ax = plt.subplots(figsize = (8, 6))
    scatter = ax.scatter(
        x, y,
        c      = residuals,
        cmap   = "coolwarm",
        s      = 25,
        alpha  = 0.8,
        zorder = 3,
    )
    ax.plot(
        x_line, y_line,
        color     = "black",
        linewidth = 1.5,
        label     = f"OLS  r = {r_val:.3f},  p = {fmt_p(p_val)}",
    )
    ax.plot(
        x_line, y_bayes,
        color     = "green",
        linewidth = 1.5,
        label     = "Bayesian ridge (estimated priors)",
    )
    ax.fill_between(
        x_line,
        hpdi_low,
        hpdi_high,
        color = "green",
        alpha = 0.15,
        label = "94% HPDI",
    )
    ax.set_xlabel("Pearson r to Template")
    ax.set_ylabel("Mean Pearson r to Other Targets")
    ax.set_title("Template Alignment vs Target Cohesion (Chapter 7)")
    ax.legend(fontsize = 9)
    ax.grid(alpha = 0.3)

    cbar = fig.colorbar(scatter, ax = ax, fraction = 0.046, pad = 0.04)
    cbar.set_label("Residual")

    fig.subplots_adjust(left = 0.12, right = 0.87, bottom = 0.10, top = 0.93)
    fig.savefig(output_path / "ch7_template_vs_cohesion.png", dpi = 200)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Chapter 6 and Chapter 7 analyses."""
    cur_path    = Path(__file__).parent.resolve()
    output_path = cur_path / "outputs" / "pearson" / "target-cohesion"
    output_path.mkdir(parents = True, exist_ok = True)

    ranked_df = load_ranked_targets(cur_path)

    # Compute per-target mean r to the rest of the full target set.
    #   Used by both chapters.
    waveforms         = np.vstack(ranked_df["waveform"].to_list())
    full_corr         = np.corrcoef(waveforms)
    n                 = waveforms.shape[0]
    diag_mask         = np.eye(n, dtype = bool)
    full_corr_no_diag = np.where(diag_mask, np.nan, full_corr)
    per_target_mean_r = np.nanmean(full_corr_no_diag, axis = 1)

    ranked_df = ranked_df.copy()
    ranked_df["mean_r_to_others"] = per_target_mean_r

    chapter_6_cohesion_by_rank(ranked_df, output_path)
    chapter_7_template_vs_cohesion(ranked_df, output_path)


if __name__ == "__main__":
    main()
