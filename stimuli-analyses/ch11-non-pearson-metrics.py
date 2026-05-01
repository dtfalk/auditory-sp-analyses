"""Chapter 11. Choosing Alternative Metrics.

Goal
----
Define a small, principled set of non-Pearson similarity metrics, compute the
full 300x300 pairwise similarity matrix for each, and cache them on disk so
chapters 12-14 can reuse them without recomputation.

Selected metrics (each returns a similarity in roughly [-1, 1] or [0, 1])
------------------------------------------------------------------------
1. pearson_raw        - Pearson r on centered raw waveforms.  REFERENCE METRIC
                        (this is what Part II analyzed); included so downstream
                        chapters can compare directly without rerunning.

2. lagged_xcorr_max   - Maximum absolute value of the normalized cross-
                        correlation between waveform a and waveform b over
                        lags within +/- 200 ms.  Removes the strict alignment
                        requirement of Pearson: two clips that share the same
                        local pattern but at slightly different times will
                        score high here, low under Pearson.

3. envelope_corr      - Pearson r on the smoothed Hilbert amplitude envelopes
                        (low-passed at ~50 Hz).  Captures slow energy contour
                        agreement (loud-quiet pattern) while discarding
                        carrier-phase detail.

4. log_spectrum_corr  - Pearson r on log(1 + |rFFT|) spectra of the centered
                        waveforms.  Translation invariant in time, captures
                        agreement of long-term frequency content.

5. mfcc_cosine        - Cosine similarity between mean-pooled MFCC vectors
                        (n_mfcc = 13, default librosa params).  A standard
                        speech-oriented descriptor that compresses the spectral
                        envelope into a perceptually motivated low-dimensional
                        feature.

Disclaimer
----------
These are *exploratory structural* metrics over the audio waveforms.  They are
not claims about what perceptual dimensions subjects used.  The aim is to ask
whether alternative structural lenses reveal stronger grouping than Pearson.

Outputs
-------
outputs/non_pearson/ch11-metrics/
    similarity_matrices.npz   - dict-like file:
        pearson_raw         (300, 300) float32
        lagged_xcorr_max    (300, 300) float32
        envelope_corr       (300, 300) float32
        log_spectrum_corr   (300, 300) float32
        mfcc_cosine         (300, 300) float32
        labels              (300,)     <U10  ('target' / 'distractor')
        stimulus_numbers    (300,)     int64
    metric_definitions.csv    - one row per metric: name, family, range, notes.
"""

from pathlib import Path
import numpy as np
from scipy.io     import wavfile
from scipy.signal import hilbert, correlate
import pandas as pd
import librosa


# ============================================================================
# Helpers
# ============================================================================

def load_stimuli(cur_path):
    """Load all target and distractor waveforms (raw on-disk PCM, centered).

    Args:
        cur_path: Path to the script directory.

    Returns:
        Tuple (waveform_matrix, labels, numbers, sr) where waveform_matrix is
        (300, L) float64 centered, labels is np.array of strings, numbers is
        np.array of int stimulus numbers, and sr is sample rate.
    """
    stimuli_path  = cur_path / ".." / "raw_data" / "audio_stimuli"
    fs_targs_path = stimuli_path / "full_sentence"     / "targets"
    is_targs_path = stimuli_path / "imagined_sentence" / "targets"
    fs_dists_path = stimuli_path / "full_sentence"     / "distractors"
    is_dists_path = stimuli_path / "imagined_sentence" / "distractors"

    rows    = []
    labels  = []
    numbers = []
    sr_seen = None

    for label, directory in [("target", fs_targs_path),
                             ("target", is_targs_path),
                             ("distractor", fs_dists_path),
                             ("distractor", is_dists_path)]:
        for wav_path in sorted(directory.glob("*.wav")):
            sr, raw = wavfile.read(wav_path)
            if sr_seen is None:
                sr_seen = sr
            assert sr == sr_seen, f"Sample rate mismatch: {sr} vs {sr_seen} in {wav_path}"
            rows.append(raw.astype(np.float64))
            labels.append(label)
            numbers.append(int(wav_path.stem))

    waveform_matrix = np.vstack(rows)
    waveform_matrix = waveform_matrix - waveform_matrix.mean(axis = 1, keepdims = True)

    print(f"Loaded {waveform_matrix.shape[0]} stimuli, length L={waveform_matrix.shape[1]} "
          f"samples, sr={sr_seen} Hz (raw on-disk, centered).", flush = True)
    return waveform_matrix, np.array(labels), np.array(numbers, dtype = np.int64), sr_seen


# ============================================================================
# Metric 1: Pearson r (reference)
# ============================================================================

def metric_pearson(waveform_matrix):
    """Pearson r on centered waveforms.

    Args:
        waveform_matrix: (n, L) centered array.

    Returns:
        (n, n) float32 similarity matrix.
    """
    return np.corrcoef(waveform_matrix).astype(np.float32)


# ============================================================================
# Metric 2: Lagged cross-correlation maximum
# ============================================================================

def metric_lagged_xcorr(waveform_matrix, sr, max_lag_ms = 200.0):
    """Max-absolute normalized cross-correlation within +/- max_lag_ms.

    For each pair (i, j) computes:
        x_ij(tau) = sum_t a_i[t] * a_j[t + tau] / (||a_i|| * ||a_j||)
    and reports max_|tau| <= max_lag |x_ij(tau)|.

    Implementation uses the FFT once per stimulus to make the all-pairs cost
    O(n^2 * F log F) where F = next_fast_len(2L - 1).

    Args:
        waveform_matrix: (n, L) centered array.
        sr:              Sample rate (Hz).
        max_lag_ms:      Half-window of lags to consider, in milliseconds.

    Returns:
        (n, n) float32 similarity matrix in [0, 1].
    """
    from scipy.fft  import rfft, irfft, next_fast_len

    n, L         = waveform_matrix.shape
    norms        = np.linalg.norm(waveform_matrix, axis = 1)
    assert np.all(norms > 0), "Zero-norm signal in lagged xcorr"

    nfft         = next_fast_len(2 * L - 1)
    spec         = rfft(waveform_matrix, n = nfft, axis = 1)         # (n, nfft//2 + 1)
    spec_conj    = np.conj(spec)
    max_lag      = int(round(max_lag_ms * 1e-3 * sr))
    assert 0 < max_lag < L, f"max_lag {max_lag} out of bounds for L={L}"

    sim          = np.empty((n, n), dtype = np.float32)
    for i in range(n):
        # Cross-correlation of a_i with all a_j at once via FFT product.
        prod      = spec[i][np.newaxis, :] * spec_conj   # (n, nfft//2 + 1)
        xcorr     = irfft(prod, n = nfft, axis = 1)      # (n, nfft)
        # Lag 0 sits at index 0; positive lags 1..L-1; negative lags wrap from end.
        pos_part  = xcorr[:, 1:max_lag + 1]              # lags +1 .. +max_lag
        neg_part  = xcorr[:, -max_lag:]                  # lags -max_lag .. -1
        zero_lag  = xcorr[:, 0:1]                        # lag 0
        window    = np.concatenate([neg_part, zero_lag, pos_part], axis = 1)

        denom     = norms[i] * norms                     # (n,)
        sim_row   = np.max(np.abs(window), axis = 1) / denom
        sim[i, :] = sim_row.astype(np.float32)

        if (i + 1) % 50 == 0:
            print(f"    lagged_xcorr   row {i + 1}/{n}", flush = True)

    # Symmetrize (numerical noise from independent rows).
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)
    return sim


# ============================================================================
# Metric 3: Envelope correlation
# ============================================================================

def metric_envelope_corr(waveform_matrix, sr, smooth_hz = 50.0):
    """Pearson r on smoothed Hilbert amplitude envelopes.

    Smoothing is a centered moving-average filter of width ~ sr / smooth_hz
    samples, which acts as a crude low-pass at ~smooth_hz.

    Args:
        waveform_matrix: (n, L) array (centering not required for envelope).
        sr:              Sample rate (Hz).
        smooth_hz:       Approximate envelope low-pass cutoff in Hz.

    Returns:
        (n, n) float32 similarity matrix in [-1, 1].
    """
    n, L            = waveform_matrix.shape
    analytic        = hilbert(waveform_matrix, axis = 1)
    envelope        = np.abs(analytic)

    win             = max(3, int(round(sr / smooth_hz)))
    if win % 2 == 0:
        win += 1
    kernel          = np.ones(win, dtype = np.float64) / win
    # Apply moving average per row via FFT-free convolution (small window).
    smoothed        = np.empty_like(envelope)
    for i in range(n):
        smoothed[i] = np.convolve(envelope[i], kernel, mode = "same")

    print(f"    envelope_corr  smoothing window = {win} samples (~{sr / win:.1f} Hz)",
          flush = True)
    return np.corrcoef(smoothed).astype(np.float32)


# ============================================================================
# Metric 4: Log-power-spectrum correlation
# ============================================================================

def metric_log_spectrum_corr(waveform_matrix):
    """Pearson r between log(1 + |rFFT|) magnitude spectra.

    Args:
        waveform_matrix: (n, L) centered array.

    Returns:
        (n, n) float32 similarity matrix in [-1, 1].
    """
    spec     = np.abs(np.fft.rfft(waveform_matrix, axis = 1))
    log_spec = np.log1p(spec)
    return np.corrcoef(log_spec).astype(np.float32)


# ============================================================================
# Metric 5: MFCC cosine similarity
# ============================================================================

def metric_mfcc_cosine(waveform_matrix, sr, n_mfcc = 13):
    """Cosine similarity between MFCC summary vectors.

    For each clip computes MFCCs (default librosa hop/window).  Drops the C0
    coefficient (which encodes log frame energy and is near-constant across
    RMS-normalized clips, drowning out the rest).  Summarizes the remaining
    n_mfcc - 1 coefficients by their per-coefficient mean and standard
    deviation across time, giving a 2 * (n_mfcc - 1)-D feature vector per clip.
    Reports cosine similarity between feature vectors.

    Args:
        waveform_matrix: (n, L) array.
        sr:              Sample rate (Hz).
        n_mfcc:          Number of MFCC coefficients per frame.

    Returns:
        (n, n) float32 similarity matrix in [-1, 1].
    """
    n        = waveform_matrix.shape[0]
    feat_dim = 2 * (n_mfcc - 1)
    feats    = np.empty((n, feat_dim), dtype = np.float64)
    for i in range(n):
        clip       = waveform_matrix[i].astype(np.float32)
        mfcc       = librosa.feature.mfcc(y = clip, sr = sr, n_mfcc = n_mfcc)
        mfcc_no_c0 = mfcc[1:, :]                                # drop C0
        mean_part  = mfcc_no_c0.mean(axis = 1)
        std_part   = mfcc_no_c0.std(axis  = 1)
        feats[i]   = np.concatenate([mean_part, std_part])
        if (i + 1) % 50 == 0:
            print(f"    mfcc_cosine    feat {i + 1}/{n}", flush = True)

    # Z-score each feature dimension across clips so no single coefficient
    # dominates the cosine.
    feats    = (feats - feats.mean(axis = 0)) / (feats.std(axis = 0) + 1e-12)
    norms    = np.linalg.norm(feats, axis = 1, keepdims = True)
    assert np.all(norms > 0), "Zero-norm MFCC summary vector"
    unit     = feats / norms
    return (unit @ unit.T).astype(np.float32)


# ============================================================================
# Main
# ============================================================================

def main():
    cur_path    = Path(__file__).resolve().parent
    output_path = cur_path / "outputs" / "non_pearson" / "ch11-metrics"
    output_path.mkdir(parents = True, exist_ok = True)

    waveform_matrix, labels, numbers, sr = load_stimuli(cur_path)

    print(f"Computing pearson_raw ...", flush = True)
    pearson_mat   = metric_pearson(waveform_matrix)

    print(f"Computing lagged_xcorr_max ...", flush = True)
    xcorr_mat     = metric_lagged_xcorr(waveform_matrix, sr, max_lag_ms = 200.0)

    print(f"Computing envelope_corr ...", flush = True)
    env_mat       = metric_envelope_corr(waveform_matrix, sr, smooth_hz = 50.0)

    print(f"Computing log_spectrum_corr ...", flush = True)
    spec_mat      = metric_log_spectrum_corr(waveform_matrix)

    print(f"Computing mfcc_cosine ...", flush = True)
    mfcc_mat      = metric_mfcc_cosine(waveform_matrix, sr, n_mfcc = 13)

    np.savez(output_path / "similarity_matrices.npz",
             pearson_raw         = pearson_mat,
             lagged_xcorr_max    = xcorr_mat,
             envelope_corr       = env_mat,
             log_spectrum_corr   = spec_mat,
             mfcc_cosine         = mfcc_mat,
             labels              = labels,
             stimulus_numbers    = numbers)

    pd.DataFrame([
        {"metric":  "pearson_raw",
         "family":  "time-domain",
         "range":   "[-1, 1]",
         "notes":   "Reference metric (Part II). Aligned waveform Pearson r."},
        {"metric":  "lagged_xcorr_max",
         "family":  "time-domain",
         "range":   "[0, 1]",
         "notes":   "Max |normalized cross-correlation| within +/-200 ms; relaxes alignment."},
        {"metric":  "envelope_corr",
         "family":  "envelope-based",
         "range":   "[-1, 1]",
         "notes":   "Pearson r on smoothed Hilbert envelope (low-pass ~50 Hz); slow energy contour."},
        {"metric":  "log_spectrum_corr",
         "family":  "frequency-domain",
         "range":   "[-1, 1]",
         "notes":   "Pearson r on log(1+|rFFT|); time-shift invariant frequency content."},
        {"metric":  "mfcc_cosine",
         "family":  "speech-oriented",
         "range":   "[-1, 1]",
         "notes":   "Cosine sim of z-scored mean+std-pooled MFCC(1..12) features (C0 dropped); "
                    "compressed spectral envelope summary."},
    ]).to_csv(output_path / "metric_definitions.csv", index = False)

    # Quick on-screen sanity: each metric's mean diagonal (should be ~1) and
    # mean off-diagonal.
    print("Metric sanity check (diag should be ~1, off-diag varies):", flush = True)
    for name, mat in [("pearson_raw",       pearson_mat),
                      ("lagged_xcorr_max",  xcorr_mat),
                      ("envelope_corr",     env_mat),
                      ("log_spectrum_corr", spec_mat),
                      ("mfcc_cosine",       mfcc_mat)]:
        diag = float(np.diag(mat).mean())
        off  = mat[np.triu_indices_from(mat, k = 1)]
        print(f"  {name:20s}  diag={diag:+.4f}  off-diag mean={float(off.mean()):+.4f}  "
              f"std={float(off.std()):.4f}", flush = True)

    print(f"Chapter 11 done.  Outputs saved to: {output_path}", flush = True)


if __name__ == "__main__":
    main()
