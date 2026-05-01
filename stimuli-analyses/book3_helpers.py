"""Book 3 shared loaders.

Adds waveform loading on top of Book 2 helpers.
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy   as np
import pandas  as pd

from scipy.io import wavfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book2_helpers import (load_all_trials, build_stimulus_block_map,
                           load_similarity_matrices,
                           build_subject_stim_matrix, RAW_PATH)

# Re-export for convenience
__all__ = ["load_all_trials", "build_stimulus_block_map",
           "load_similarity_matrices", "build_subject_stim_matrix",
           "load_block_waveforms", "load_template", "load_block_order",
           "load_questionnaires", "RAW_PATH"]

AUDIO_PATH = RAW_PATH / "audio_stimuli"


# ============================================================================
# Block order + questionnaires
# ============================================================================

def _read_summary(path):
    df = pd.read_csv(path)
    row = df.iloc[0]
    return dict(subject = str(int(row["Subject Number"])),
                block_1_scheme = row["Block 1"],
                block_1_dprime = float(row["Block 1 D-Prime"]),
                block_2_scheme = row["Block 2"],
                block_2_dprime = float(row["Block 2 D-Prime"]))


def load_block_order():
    """Return DataFrame: site, subject, block_1_scheme, block_2_scheme,
    and per-block which-was-first flag.

    Long form columns: site, subject, block_type, block_order_index (1 or 2),
    summary_dprime.
    """
    rows = []
    for site in ("Oberlin", "UChicago"):
        site_root = RAW_PATH / site
        for subj_dir in sorted(p for p in site_root.iterdir() if p.is_dir()):
            subj_pad  = subj_dir.name
            sums = list(subj_dir.glob(f"summary_data_{subj_pad}.csv"))
            if not sums:
                continue
            s = _read_summary(sums[0])
            rows.append(dict(site = site, subject = s["subject"],
                             block_type        = s["block_1_scheme"],
                             block_order_index = 1,
                             summary_dprime    = s["block_1_dprime"]))
            rows.append(dict(site = site, subject = s["subject"],
                             block_type        = s["block_2_scheme"],
                             block_order_index = 2,
                             summary_dprime    = s["block_2_dprime"]))
    return pd.DataFrame(rows)


# Questionnaire definitions: (file_stem, n_items, score_name, reverse_indices)
QUESTIONNAIRES = [
    ("bais_v",                   14, "BAIS_V",          []),  # vividness
    ("bais_c",                   14, "BAIS_C",          []),  # control
    ("vhq",                      14, "VHQ",             []),
    ("tellegen",                 34, "TAS",             []),  # absorption
    ("launay_slade",             16, "LSHS",            []),  # hallucination prone
    ("dissociative_experiences", 28, "DES",             []),
    ("flow_state_scale",          9, "FSS",             []),
]


def _read_questionnaire(path, n_items):
    """First row only, sum of Q1..Qn after coercing to numeric."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    cols = [f"Q{i}" for i in range(1, n_items + 1)]
    vals = pd.to_numeric(row[cols], errors = "coerce")
    return dict(total = float(vals.sum()),
                mean  = float(vals.mean()),
                n_valid = int(vals.notna().sum()))


def load_questionnaires():
    """Return wide DataFrame: site, subject + one column per questionnaire
    (TOTAL score) plus *_mean.  Sleepiness handled separately."""
    rows = []
    for site in ("Oberlin", "UChicago"):
        site_root = RAW_PATH / site
        for subj_dir in sorted(p for p in site_root.iterdir() if p.is_dir()):
            subj_pad = subj_dir.name
            entry = dict(site = site, subject = str(int(subj_pad)))
            for stem, n, name, _ in QUESTIONNAIRES:
                f = subj_dir / f"{stem}_{subj_pad}.csv"
                if not f.exists():
                    entry[name] = np.nan
                    continue
                r = _read_questionnaire(f, n)
                entry[name] = r["total"]
                entry[f"{name}_mean"] = r["mean"]
            # sleepiness: pre/post per block
            ss_path = subj_dir / f"stanford_sleepiness_{subj_pad}.csv"
            if ss_path.exists():
                ss = pd.read_csv(ss_path)
                ss["response"] = pd.to_numeric(ss["response"], errors = "coerce")
                entry["SSS_mean"] = float(ss["response"].mean())
                entry["SSS_pre_b1"] = float(
                    ss[(ss["block_index"] == 1) & (ss["pre_or_post"] == "pre")]
                    ["response"].mean())
                entry["SSS_post_b2"] = float(
                    ss[(ss["block_index"] == 2) & (ss["pre_or_post"] == "post")]
                    ["response"].mean())
            rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================================
# Audio loaders
# ============================================================================

def load_template():
    """Return centred wall.wav as (L,) float64 + sample rate."""
    sr, raw = wavfile.read(AUDIO_PATH / "wall.wav")
    t = raw.astype(np.float64)
    t = t - t.mean()
    return t, sr


def load_block_waveforms(block_type):
    """Return (W, stim_nums, labels, sr) for the 150 stimuli of one block.
    W is (150, L) centred float64; labels in {0=distractor, 1=target}.
    Order matches sorted([targets..., distractors...]).
    """
    targs = sorted((AUDIO_PATH / block_type / "targets").glob("*.wav"))
    dists = sorted((AUDIO_PATH / block_type / "distractors").glob("*.wav"))
    rows, nums, lbls = [], [], []
    sr_seen = None
    for path in targs:
        sr, raw = wavfile.read(path)
        sr_seen = sr_seen or sr
        rows.append(raw.astype(np.float64))
        nums.append(int(path.stem))
        lbls.append(1)
    for path in dists:
        sr, raw = wavfile.read(path)
        rows.append(raw.astype(np.float64))
        nums.append(int(path.stem))
        lbls.append(0)
    W = np.vstack(rows)
    W = W - W.mean(axis = 1, keepdims = True)
    return W, np.array(nums, dtype = np.int64), np.array(lbls, dtype = np.int8), sr_seen


def magnitude_spectrum(W):
    """Return real-FFT magnitude spectrum of each row, shape (n, F)."""
    return np.abs(np.fft.rfft(W, axis = 1))
