"""Shared loaders for Book 2 (subject-behaviour) chapters.

Everything here is pure I/O + light reshaping so each chapter script can stay
focused on its own analysis.  All trials are loaded once; per-block and
per-subject views are built on demand.
"""

# ============================================================================
# Imports and paths
# ============================================================================

from pathlib import Path

import numpy  as np
import pandas as pd

PKG_PATH = Path(__file__).resolve().parent
RAW_PATH = PKG_PATH / ".." / "raw_data"
SIM_NPZ  = PKG_PATH / "outputs" / "non_pearson" / "ch11-metrics" / "similarity_matrices.npz"

METRICS = ["pearson_raw", "lagged_xcorr_max", "envelope_corr",
           "log_spectrum_corr", "mfcc_cosine"]


# ============================================================================
# Stimulus -> block_type map (from on-disk audio layout)
# ============================================================================

def build_stimulus_block_map():
    """Return dict {stimulus_number: block_type}."""
    base = RAW_PATH / "audio_stimuli"
    out  = {}
    for block in ("full_sentence", "imagined_sentence"):
        for kind in ("targets", "distractors"):
            for wav in (base / block / kind).glob("*.wav"):
                num = int(wav.stem)
                assert num not in out, f"Duplicate stimulus {num}"
                out[num] = block
    return out


# ============================================================================
# Trial loader
# ============================================================================

def load_all_trials():
    """Load every trial across both sites, both blocks.

    Returns DataFrame with columns:
        site, subject, block_type, stimulus_number, trial_index,
        true_label, subject_response, called_target, true_target
    where trial_index is the within-(subject, block) order (0-indexed).
    """
    rows = []
    for site in ("Oberlin", "UChicago"):
        site_root = RAW_PATH / site
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
                df["trial_index"]     = np.arange(len(df), dtype = np.int64)
                rows.append(df)
    out = pd.concat(rows, ignore_index = True)
    assert set(out["true_label"].unique())       <= {"target", "distractor"}
    assert set(out["subject_response"].unique()) <= {"target", "distractor"}
    return out


# ============================================================================
# Similarity matrix loader (from ch11)
# ============================================================================

def load_similarity_matrices():
    """Return dict of metric -> (sim_matrix, stim_numbers, labels)."""
    npz = np.load(SIM_NPZ, allow_pickle = True)
    stim_nums = npz["stimulus_numbers"].astype(np.int64)
    labels    = np.asarray(npz["labels"])
    out = {}
    for m in METRICS:
        out[m] = (np.asarray(npz[m]), stim_nums, labels)
    return out


# ============================================================================
# Per-block subject x stimulus response matrix
# ============================================================================

def build_subject_stim_matrix(trials, block_type):
    """For one block, return:
        Y           : (n_subjects, n_stimuli) called-target matrix (NaN if missing)
        subjects    : list of subject ids
        stim_nums   : np.array of stimulus_number for columns
        true_labels : np.array of 0/1 true_target aligned with columns
    """
    sub_block = trials[trials["block_type"] == block_type].copy()
    subjects  = sorted(sub_block["subject"].unique())
    stim_nums = np.array(sorted(sub_block["stimulus_number"].unique()),
                         dtype = np.int64)
    s2idx = {s: i for i, s in enumerate(subjects)}
    n2idx = {n: j for j, n in enumerate(stim_nums)}
    Y = np.full((len(subjects), len(stim_nums)), np.nan, dtype = np.float64)
    for r in sub_block.itertuples(index = False):
        Y[s2idx[r.subject], n2idx[r.stimulus_number]] = r.called_target
    true_labels = np.zeros(len(stim_nums), dtype = np.int64)
    for r in sub_block.itertuples(index = False):
        true_labels[n2idx[r.stimulus_number]] = r.true_target
    return Y, subjects, stim_nums, true_labels


# ============================================================================
# Helper: per-stimulus feature matrix aligned to a block
# ============================================================================

def block_feature_table(trials, block_map, sim_dict, block_type):
    """Build a per-stimulus feature row for one block.

    For each stimulus i in this block and each metric m, we compute the
    "block-internal target-vs-distractor similarity contrast":

        f_m(i) = mean_{j in block, true=target, j != i} sim_m(i, j)
               - mean_{j in block, true=distract, j != i} sim_m(i, j)

    Positive f_m means stimulus i looks (under metric m) more like the block's
    other targets than the block's other distractors.

    Returns DataFrame with columns:
        stimulus_number, true_target, p_called_target, n_trials, <metric>_contrast
    """
    sub = trials[trials["block_type"] == block_type]
    per_stim = (sub.groupby("stimulus_number", as_index = False)
                   .agg(n_trials        = ("called_target", "size"),
                        p_called_target = ("called_target", "mean"),
                        true_target     = ("true_target",   "first")))
    # restrict everything to stims present in this block
    per_stim["stimulus_number"] = per_stim["stimulus_number"].astype(np.int64)
    keep = np.array([block_map.get(int(n)) == block_type for n in per_stim["stimulus_number"]])
    per_stim = per_stim[keep].reset_index(drop = True)
    block_stims = per_stim["stimulus_number"].to_numpy()
    for metric, (sim, stim_nums, labels) in sim_dict.items():
        idx = {int(n): k for k, n in enumerate(stim_nums)}
        block_idx     = np.array([idx[int(n)] for n in block_stims])
        block_labels  = np.array([labels[k] for k in block_idx])
        is_t = block_labels == "target"
        is_d = block_labels == "distractor"
        sub_sim = sim[np.ix_(block_idx, block_idx)]
        contrast = np.zeros(len(block_stims))
        for k in range(len(block_stims)):
            row = sub_sim[k]
            mask_t = is_t.copy(); mask_t[k] = False
            mask_d = is_d.copy(); mask_d[k] = False
            mu_t = row[mask_t].mean() if mask_t.any() else 0.0
            mu_d = row[mask_d].mean() if mask_d.any() else 0.0
            contrast[k] = mu_t - mu_d
        per_stim[f"{metric}_contrast"] = contrast
    return per_stim
