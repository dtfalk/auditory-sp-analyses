from pathlib import Path
import numpy as np
import logging
from scipy.io import wavfile
from scipy.stats import pearsonr 
import pandas as pd
import matplotlib.pyplot as plt


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


def main():

    # Current directory
    cur_path = Path(__file__).parent.resolve()
    
    # Output folders
    summary_results_path     = cur_path / "outputs" / "pearson" / "pariwise" / "summary"
    all_data_results_path    = cur_path / "outputs" / "pearson" / "pariwise" / "all_data"
    summary_results_path.mkdir(parents = True, exist_ok = True)
    all_data_results_path.mkdir(parents = True, exist_ok = True)
    
    # Get the stimuli 
    stimuli = get_stimuli(cur_path)

    # Verify all stimuli propertes are as expected
    verify_stimuli_properties(stimuli)
    

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
    tt_vals = tt[np.triu_indices_from(tt, k=1)]
    dd_vals = dd[np.triu_indices_from(dd, k=1)]

    # For cross condition, take top-right block
    n_t = targ_matrix.shape[0]
    n_d = dist_matrix.shape[0]
    td_vals = td[:n_t, n_t:n_t+n_d].flatten()

    # Save raw data
    pd.DataFrame(tt_vals, columns=["pearson_r"]).to_csv(
        all_data_results_path / "target_target_raw.csv", index=False
    )
    pd.DataFrame(dd_vals, columns=["pearson_r"]).to_csv(
        all_data_results_path / "distractor_distractor_raw.csv", index=False
    )
    pd.DataFrame(td_vals, columns=["pearson_r"]).to_csv(
        all_data_results_path / "target_distractor_raw.csv", index=False
    )

    # Compute stats
    tt_mean = np.mean(tt_vals)
    tt_std  = np.std(tt_vals)
    tt_se   = tt_std / np.sqrt(len(tt_vals))

    dd_mean = np.mean(dd_vals)
    dd_std  = np.std(dd_vals)
    dd_se   = dd_std / np.sqrt(len(dd_vals))

    td_mean = np.mean(td_vals)
    td_std  = np.std(td_vals)
    td_se   = td_std / np.sqrt(len(td_vals))

    # Save summary
    summary_df = pd.DataFrame([
        ["target-target", tt_mean, tt_std, tt_se, len(tt_vals)],
        ["distractor-distractor", dd_mean, dd_std, dd_se, len(dd_vals)],
        ["target-distractor", td_mean, td_std, td_se, len(td_vals)],
    ], columns=["condition", "mean", "std", "se", "n"])

    summary_df.to_csv(summary_results_path / "summary_stats.csv", index=False)

    # Plot
    plt.figure()
    plt.bar(
        summary_df["condition"],
        summary_df["mean"],
        yerr=summary_df["se"],
        capsize=5
    )
    plt.ylabel("Pearson r")
    plt.title("Pairwise Correlations")
    plt.tight_layout()
    plt.savefig(summary_results_path / "pearson_barplot.png")
    plt.close()
        
    
    
        
 
if __name__ == "__main__":
    main()