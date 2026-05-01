"""Chapter 15.  Subject response prediction.

Goal
----
Move from "what structure is in the stimuli" to "what structure did subjects
actually use".  We do not assume the experimenter labels (target/distractor)
are what subjects' decisions track.  For each similarity metric M (Pearson
raw, lagged xcorr, envelope, log spectrum, MFCC) we ask:

    1. How well does M-similarity-to-the-block-target predict, per stimulus,
       the rate at which subjects called that stimulus a target?
    2. At trial level, how well does M predict the binary subject response,
       compared to the experimenter's true label?
    3. Do M's label-blind k=2 clusters (from ch13) agree with subject
       responses better, worse, or differently than the experimenter labels?

This makes "labels that aren't ours" a feature, not a failure mode: a metric
whose blind clustering disagrees with our labels but agrees with subject
responses is more informative about subject behavior than the labels
themselves.

Inputs
------
- stimuli-analyses/outputs/non_pearson/ch11-metrics/similarity_matrices.npz
- stimuli-analyses/outputs/non_pearson/ch13-label-blind/ch13_assignments_*.csv
- raw_data/{Oberlin,UChicago}/<subject>/<block>/<block>_<subject>.csv
- raw_data/audio_stimuli/{full_sentence,imagined_sentence}/{targets,distractors}/*.wav
  (used only to recover the per-stimulus block_type via filename location)

Outputs
-------
outputs/pearson/ch15-subject-prediction/
    ch15_per_stimulus_response.csv
    ch15_per_metric_stimulus_features.csv
    ch15_metric_stimulus_level_prediction.csv
    ch15_metric_trial_level_prediction.csv
    ch15_metric_vs_label_agreement.csv
    ch15_subject_summary.csv
    ch15_metric_stimulus_r.png
    ch15_metric_trial_auc.png
    ch15_response_agreement.png
"""

# ============================================================================
# Imports and setup
# ============================================================================

from pathlib import Path

import numpy            as np
import pandas           as pd
import matplotlib.pyplot as plt

from scipy.io      import wavfile
from scipy.stats   import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import roc_auc_score

cur_path = Path(__file__).resolve().parent
out_path = cur_path / "outputs" / "pearson" / "ch15-subject-prediction"
out_path.mkdir(parents = True, exist_ok = True)


# ============================================================================
# Stimulus -> block_type map (from on-disk layout)
# ============================================================================

def build_stimulus_block_map():
    """Return dict {stimulus_number: block_type} where block_type is in
    {'full_sentence', 'imagined_sentence'}.

    Walks raw_data/audio_stimuli/{full_sentence,imagined_sentence}/{targets,distractors}/.
    """
    base = cur_path / ".." / "raw_data" / "audio_stimuli"
    out  = {}
    for block in ("full_sentence", "imagined_sentence"):
        for kind in ("targets", "distractors"):
            for wav in (base / block / kind).glob("*.wav"):
                num = int(wav.stem)
                assert num not in out, f"Duplicate stimulus number {num} across blocks"
                out[num] = block
    return out


# ============================================================================
# Subject response loader
# ============================================================================

def load_all_trials():
    """Load all trial-level subject responses from Oberlin and UChicago.

    Returns:
        DataFrame columns:
            site, subject, block_type, stimulus_number,
            true_label, subject_response,
            called_target (1 if subject_response == 'target' else 0),
            true_target   (1 if true_label       == 'target' else 0).
    """
    raw_root = cur_path / ".." / "raw_data"
    rows     = []
    for site in ("Oberlin", "UChicago"):
        site_root = raw_root / site
        if not site_root.exists():
            continue
        for subj_dir in sorted(site_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj = subj_dir.name
            for block in ("full_sentence", "imagined_sentence"):
                trial_csv = subj_dir / block / f"{block}_{subj}.csv"
                if not trial_csv.exists():
                    continue
                df = pd.read_csv(trial_csv)
                df = df[["Subject Number", "Block Scheme", "Stimulus Number",
                         "Stimulus Type", "Subject Response"]].copy()
                df.columns = ["subject", "block_type", "stimulus_number",
                              "true_label", "subject_response"]
                df["site"]            = site
                df["subject"]         = df["subject"].astype(str)
                df["stimulus_number"] = df["stimulus_number"].astype(np.int64)
                df["called_target"]   = (df["subject_response"] == "target").astype(int)
                df["true_target"]     = (df["true_label"]       == "target").astype(int)
                rows.append(df)
    out = pd.concat(rows, ignore_index = True)
    assert set(out["true_label"].unique())       <= {"target", "distractor"}
    assert set(out["subject_response"].unique()) <= {"target", "distractor"}
    return out


# ============================================================================
# Per-stimulus response aggregation
# ============================================================================

def aggregate_per_stimulus(trials):
    """Aggregate trial-level responses to per-stimulus rates.

    Returns one row per (stimulus_number, block_type) with n_trials and
    p_called_target plus the (single) true label.
    """
    grp = trials.groupby(["block_type", "stimulus_number"], as_index = False).agg(
        n_trials        = ("called_target", "size"),
        p_called_target = ("called_target", "mean"),
        true_target     = ("true_target",  "first"),
    )
    grp["true_label"] = np.where(grp["true_target"] == 1, "target", "distractor")
    return grp


# ============================================================================
# Per-metric per-stimulus features
# ============================================================================

def per_stimulus_features(sim_matrix, stim_numbers, labels, block_map,
                          per_stim_response):
    """Compute per-stimulus features from a similarity matrix.

    For each stimulus i in block B with true label l:
        sim_to_block_targets_excl_self     = mean_{j in B, true=target, j != i} sim[i,j]
        sim_to_block_distractors_excl_self = mean_{j in B, true=distract, j != i} sim[i,j]
        target_minus_distractor            = difference of the two
    Cosmetic: also reports n_block_targets and n_block_distractors used.

    Returns DataFrame indexed (block_type, stimulus_number) with feature columns.
    """
    block_array = np.array([block_map[int(s)] for s in stim_numbers])
    label_array = np.array(labels)

    n = len(stim_numbers)
    feat_rows = []
    for i in range(n):
        bi = block_array[i]
        same_block       = block_array == bi
        is_target        = label_array == "target"
        is_distractor    = label_array == "distractor"

        mask_targets     = same_block & is_target
        mask_distractors = same_block & is_distractor
        mask_targets[i]     = False
        mask_distractors[i] = False

        sim_t = sim_matrix[i, mask_targets].mean()     if mask_targets.sum()     > 0 else np.nan
        sim_d = sim_matrix[i, mask_distractors].mean() if mask_distractors.sum() > 0 else np.nan
        feat_rows.append({
            "stimulus_number"             : int(stim_numbers[i]),
            "block_type"                  : bi,
            "true_label"                  : label_array[i],
            "sim_to_block_targets"        : sim_t,
            "sim_to_block_distractors"    : sim_d,
            "target_minus_distractor"     : sim_t - sim_d,
            "n_block_targets_excl_self"   : int(mask_targets.sum()),
            "n_block_distractors_excl_self": int(mask_distractors.sum()),
        })
    feat = pd.DataFrame(feat_rows)
    merged = feat.merge(
        per_stim_response[["block_type", "stimulus_number",
                           "n_trials", "p_called_target"]],
        on = ["block_type", "stimulus_number"],
        how = "left",
    )
    return merged


# ============================================================================
# Stimulus-level prediction
# ============================================================================

def stimulus_level_prediction(per_metric_feats):
    """For each metric: Pearson r between target_minus_distractor and
    p_called_target, both overall and split by true_label.

    Returns a DataFrame summary.
    """
    out = []
    for metric, feat in per_metric_feats.items():
        f = feat.dropna(subset = ["p_called_target", "target_minus_distractor"])
        r_all,  p_all  = pearsonr(f["target_minus_distractor"], f["p_called_target"])
        ft = f[f["true_label"] == "target"]
        fd = f[f["true_label"] == "distractor"]
        r_t, p_t = pearsonr(ft["target_minus_distractor"], ft["p_called_target"])
        r_d, p_d = pearsonr(fd["target_minus_distractor"], fd["p_called_target"])
        out.append({
            "metric"               : metric,
            "n_stimuli"            : len(f),
            "r_all"                : r_all,
            "p_all"                : p_all,
            "r_within_targets"     : r_t,
            "p_within_targets"     : p_t,
            "r_within_distractors" : r_d,
            "p_within_distractors" : p_d,
        })
    return pd.DataFrame(out).sort_values("r_all", ascending = False).reset_index(drop = True)


# ============================================================================
# Trial-level prediction (logistic regression, leave-one-subject-out AUC)
# ============================================================================

def trial_level_prediction(trials, per_metric_feats):
    """Predict subject called_target from each metric's
    target_minus_distractor feature, plus a baseline using true_target.

    Pools all trials.  Reports:
        - within-sample AUC of a logistic fit
        - Pearson r between feature and called_target across trials
    Subjects are not held out (300 stimuli x ~25 subjects ~= 7500 trials, far
    more than the per-stimulus design matrix); the feature itself is per
    stimulus so out-of-sample at trial level is not the right unit.  See
    stimulus-level table for the cleaner separation.
    """
    out = []
    for metric, feat in per_metric_feats.items():
        merged = trials.merge(
            feat[["block_type", "stimulus_number", "target_minus_distractor"]],
            on = ["block_type", "stimulus_number"],
            how = "left",
        ).dropna(subset = ["target_minus_distractor"])
        x = merged["target_minus_distractor"].values.reshape(-1, 1)
        y = merged["called_target"].values
        clf = LogisticRegression()
        clf.fit(x, y)
        proba = clf.predict_proba(x)[:, 1]
        auc   = roc_auc_score(y, proba)
        r, p  = pearsonr(merged["target_minus_distractor"], merged["called_target"])
        out.append({
            "metric"                : metric,
            "n_trials"              : len(merged),
            "auc_logistic"          : auc,
            "r_feature_vs_called"   : r,
            "p_feature_vs_called"   : p,
        })
    # Baseline: experimenter true label
    y_all   = trials["called_target"].values
    x_all   = trials["true_target"].values
    auc_lab = roc_auc_score(y_all, x_all)
    r_lab, p_lab = pearsonr(x_all, y_all)
    out.append({
        "metric"               : "true_label_baseline",
        "n_trials"             : len(trials),
        "auc_logistic"         : auc_lab,
        "r_feature_vs_called"  : r_lab,
        "p_feature_vs_called"  : p_lab,
    })
    return pd.DataFrame(out).sort_values("auc_logistic", ascending = False).reset_index(drop = True)


# ============================================================================
# Cluster vs. response agreement
# ============================================================================

def cluster_vs_response_agreement(trials, ch13_dir, metric_keys):
    """For each metric's ch13 label-blind k=2 clusters, compute fraction of
    trials whose subject_response matches the cluster (under both possible
    cluster->response label assignments; report the better one), and compare
    to the same fraction under the true experimenter label.

    Returns DataFrame.
    """
    rows = []
    # baseline
    base_match = (trials["called_target"] == trials["true_target"]).mean()
    rows.append({"metric": "true_label_baseline",
                 "method": "experimenter",
                 "n_trials": len(trials),
                 "best_response_match_rate": base_match})

    for metric in metric_keys:
        path = ch13_dir / f"ch13_assignments_{metric}.csv"
        if not path.exists():
            continue
        assn = pd.read_csv(path)
        for method in ("agglomerative_avg", "spectral_kmeans"):
            merged = trials.merge(
                assn[["stimulus_number", method]].rename(
                    columns = {method: "cluster"}),
                on = "stimulus_number",
                how = "left",
            ).dropna(subset = ["cluster"])
            # Try both polarities: cluster 0 -> target  vs  cluster 0 -> distractor
            c = merged["cluster"].astype(int).values
            y = merged["called_target"].values
            match_a = ((c == 1).astype(int) == y).mean()  # cluster1 = target
            match_b = ((c == 0).astype(int) == y).mean()  # cluster0 = target
            rows.append({"metric": metric,
                         "method": method,
                         "n_trials": len(merged),
                         "best_response_match_rate": max(match_a, match_b)})
    return pd.DataFrame(rows).sort_values(
        "best_response_match_rate", ascending = False).reset_index(drop = True)


# ============================================================================
# Per-subject summary
# ============================================================================

def subject_summary(trials):
    """Per-subject hit/false-alarm/d-prime-ish stats and overall agreement
    with experimenter labels.
    """
    out = []
    for (site, subj, block), g in trials.groupby(["site", "subject", "block_type"]):
        n_t   = (g["true_target"] == 1).sum()
        n_d   = (g["true_target"] == 0).sum()
        hits  = ((g["true_target"] == 1) & (g["called_target"] == 1)).sum()
        fa    = ((g["true_target"] == 0) & (g["called_target"] == 1)).sum()
        out.append({
            "site"           : site,
            "subject"        : subj,
            "block_type"     : block,
            "n_target"       : int(n_t),
            "n_distractor"   : int(n_d),
            "hits"           : int(hits),
            "false_alarms"   : int(fa),
            "hit_rate"       : hits / n_t if n_t else np.nan,
            "false_alarm_rate": fa  / n_d if n_d else np.nan,
            "label_match_rate": (g["called_target"] == g["true_target"]).mean(),
        })
    return pd.DataFrame(out)


# ============================================================================
# Plots
# ============================================================================

def plot_stimulus_level(stim_summary, fig_path):
    fig, ax = plt.subplots(figsize = (7, 4))
    s = stim_summary.sort_values("r_all")
    ax.barh(s["metric"], s["r_all"], color = "steelblue")
    ax.axvline(0, color = "black", lw = 0.5)
    ax.set_xlabel("Pearson r: per-stimulus (target_minus_distractor) vs p(called_target)")
    ax.set_title("Stimulus-level prediction of subject response rate")
    fig.tight_layout()
    fig.savefig(fig_path, dpi = 150)
    plt.close(fig)


def plot_trial_level(trial_summary, fig_path):
    fig, ax = plt.subplots(figsize = (7, 4))
    s = trial_summary.sort_values("auc_logistic")
    colors = ["firebrick" if m == "true_label_baseline" else "steelblue"
              for m in s["metric"]]
    ax.barh(s["metric"], s["auc_logistic"], color = colors)
    ax.axvline(0.5, color = "black", lw = 0.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC predicting trial-level called_target")
    ax.set_title("Trial-level prediction of subject responses")
    fig.tight_layout()
    fig.savefig(fig_path, dpi = 150)
    plt.close(fig)


def plot_response_agreement(agree, fig_path):
    fig, ax = plt.subplots(figsize = (7, 5))
    s = agree.copy()
    s["tag"] = s["metric"] + " | " + s["method"]
    s = s.sort_values("best_response_match_rate")
    colors = ["firebrick" if m == "true_label_baseline" else "steelblue"
              for m in s["metric"]]
    ax.barh(s["tag"], s["best_response_match_rate"], color = colors)
    ax.axvline(0.5, color = "black", lw = 0.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Best-polarity trial-level agreement with subject response")
    ax.set_title("Cluster (label-blind) vs. experimenter label, vs. subjects")
    fig.tight_layout()
    fig.savefig(fig_path, dpi = 150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading similarity matrices ...", flush = True)
    sim_npz = np.load(cur_path / "outputs" / "non_pearson" / "ch11-metrics"
                      / "similarity_matrices.npz", allow_pickle = True)
    metric_keys  = ["pearson_raw", "lagged_xcorr_max", "envelope_corr",
                    "log_spectrum_corr", "mfcc_cosine"]
    stim_numbers = sim_npz["stimulus_numbers"]
    labels       = sim_npz["labels"]

    block_map = build_stimulus_block_map()
    assert set(stim_numbers.tolist()) <= set(block_map.keys()), \
        "Stimuli in matrix not found on disk"

    print("Loading subject trials ...", flush = True)
    trials = load_all_trials()
    print(f"  {len(trials)} trials from {trials['subject'].nunique()} subjects "
          f"across {trials['site'].nunique()} sites.", flush = True)

    # Sanity: every trial's stimulus appears in the cached matrix
    matrix_set = set(int(s) for s in stim_numbers)
    n_unknown  = (~trials["stimulus_number"].isin(matrix_set)).sum()
    print(f"  Trials with stimulus not in matrix: {n_unknown}", flush = True)
    trials = trials[trials["stimulus_number"].isin(matrix_set)].copy()

    per_stim = aggregate_per_stimulus(trials)
    per_stim.to_csv(out_path / "ch15_per_stimulus_response.csv", index = False)
    print(f"  Per-stimulus rows: {len(per_stim)}", flush = True)

    print("Computing per-metric per-stimulus features ...", flush = True)
    per_metric_feats = {}
    feat_long_rows   = []
    for metric in metric_keys:
        sim = sim_npz[metric].astype(np.float64)
        feat = per_stimulus_features(sim, stim_numbers, labels, block_map, per_stim)
        per_metric_feats[metric] = feat
        f = feat.copy()
        f["metric"] = metric
        feat_long_rows.append(f)
    pd.concat(feat_long_rows, ignore_index = True).to_csv(
        out_path / "ch15_per_metric_stimulus_features.csv", index = False)

    print("Stimulus-level prediction ...", flush = True)
    stim_summary = stimulus_level_prediction(per_metric_feats)
    stim_summary.to_csv(out_path / "ch15_metric_stimulus_level_prediction.csv",
                        index = False)
    print(stim_summary.to_string(index = False), flush = True)

    print("Trial-level prediction ...", flush = True)
    trial_summary = trial_level_prediction(trials, per_metric_feats)
    trial_summary.to_csv(out_path / "ch15_metric_trial_level_prediction.csv",
                         index = False)
    print(trial_summary.to_string(index = False), flush = True)

    print("Cluster vs. response agreement ...", flush = True)
    ch13_dir = cur_path / "outputs" / "non_pearson" / "ch13-label-blind"
    agree    = cluster_vs_response_agreement(trials, ch13_dir, metric_keys)
    agree.to_csv(out_path / "ch15_metric_vs_label_agreement.csv", index = False)
    print(agree.to_string(index = False), flush = True)

    print("Per-subject summary ...", flush = True)
    subj = subject_summary(trials)
    subj.to_csv(out_path / "ch15_subject_summary.csv", index = False)
    print(f"  Mean label_match_rate: {subj['label_match_rate'].mean():.4f}",
          flush = True)
    print(f"  Mean hit_rate:         {subj['hit_rate'].mean():.4f}",
          flush = True)
    print(f"  Mean false_alarm_rate: {subj['false_alarm_rate'].mean():.4f}",
          flush = True)

    plot_stimulus_level(stim_summary,  out_path / "ch15_metric_stimulus_r.png")
    plot_trial_level(trial_summary,    out_path / "ch15_metric_trial_auc.png")
    plot_response_agreement(agree,     out_path / "ch15_response_agreement.png")

    print(f"Chapter 15 done.  Outputs saved to: {out_path}", flush = True)


if __name__ == "__main__":
    main()
