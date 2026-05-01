"""Chapter 31.  Capable-subject deep dive.

Premise (user-defined):  the task is hard; only some people can do it.
Subjects who cannot are not informative about *what* the people who
can are doing.  So define "capable" subsets at several thresholds in
the full_sentence block (the only block with a real group d' signal),
then characterise:

    1. their group-level performance vs. zero (one block, one
       directional test per subset and per metric -> we report the
       p-value but flag the obvious circularity for d');
    2. how they perform on the OTHER block (imagined_sentence) --
       this is NOT circular: does within-subject capability transfer?
    3. their per-subject permutation significance counts and overlap;
    4. their questionnaire profile vs. the "incapable" subjects
       (Mann-Whitney, with full descriptive table);
    5. Spearman correlations within the capable subset between
       BAIS_V / BAIS_C / TAS / LSHS / DES / FSS / SSS_mean and the
       four signal metrics.

Subset definitions (all on full_sentence):
    A: d' > 0
    B: d' >= 0.10
    C: d' top tertile (highest 9 subjects)
    D: stably_aligned   (cos_ci_true > 0 AND cos_w_true > 0
                         AND beta_truetarget > 0)
    E: any per-subject permutation p<0.05 in any of the three tests

Outputs
-------
outputs/book3/ch31-capable-subjects/
    ch31_subset_membership.csv
    ch31_subset_overlap.csv
    ch31_subset_perf_within_full.csv
    ch31_subset_perf_imagined_transfer.csv
    ch31_subset_questionnaire_means.csv
    ch31_subset_questionnaire_mwu.csv
    ch31_subset_spearman_within.csv
    ch31_subset_size_summary.csv
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy  as np
import pandas as pd

from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book3_helpers import load_questionnaires

CUR_PATH = Path(__file__).resolve().parent
OUT_PATH = CUR_PATH / "outputs" / "book3" / "ch31-capable-subjects"
OUT_PATH.mkdir(parents = True, exist_ok = True)

QUEST_COLS = ["BAIS_V", "BAIS_C", "VHQ", "TAS", "LSHS", "DES", "FSS",
              "SSS_mean"]
PERF_COLS  = ["d_prime", "cos_ci_true", "cos_w_true", "beta_truetarget"]


# ============================================================================
# Load per-subject metrics
# ============================================================================

dp = pd.read_csv(CUR_PATH / "outputs" / "book2" / "ch16-sdt"
                  / "ch16_per_subject_sdt.csv")
ci = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch22-classification-image"
                  / "ch22_per_subject_ci.csv")
sp = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch24-spectral-logistic"
                  / "ch24_per_subject_spectral.csv")
sl = pd.read_csv(CUR_PATH / "outputs" / "book3" / "ch25-reminder-anchored"
                  / "ch25_per_subject_slopes.csv")
for d in (dp, ci, sp, sl):
    d["subject"] = d["subject"].astype(str)

merged = (dp[["subject", "site", "block_type", "d_prime", "hit_rate",
                "fa_rate", "accuracy", "criterion", "p_perm_dprime"]]
          .merge(ci[["subject", "block_type", "cos_ci_true",
                      "p_perm_pos"]]
                  .rename(columns = {"p_perm_pos": "p_perm_ci"}),
                  on = ["subject", "block_type"])
          .merge(sp[["subject", "block_type", "cos_w_true",
                      "p_perm_pos"]]
                  .rename(columns = {"p_perm_pos": "p_perm_spec"}),
                  on = ["subject", "block_type"])
          .merge(sl[["subject", "block_type", "beta_truetarget"]],
                  on = ["subject", "block_type"]))
quest = load_questionnaires()
quest["subject"] = quest["subject"].astype(str)
merged = merged.merge(quest[["subject"] + QUEST_COLS], on = "subject",
                       how = "left")

full = merged[merged["block_type"] == "full_sentence"].copy().reset_index(drop = True)
imag = merged[merged["block_type"] == "imagined_sentence"].copy().reset_index(drop = True)

assert len(full) == 26 and len(imag) == 26


# ============================================================================
# Subset definitions
# ============================================================================

dprime_top9_thresh = full["d_prime"].quantile(2/3)
sa_mask = ((full["cos_ci_true"]    > 0)
           & (full["cos_w_true"]    > 0)
           & (full["beta_truetarget"] > 0))
any_perm_mask = ((full["p_perm_dprime"] < 0.05)
                 | (full["p_perm_ci"]   < 0.05)
                 | (full["p_perm_spec"] < 0.05))

subsets = {
    "A_dprime_pos"      : full["d_prime"] > 0,
    "B_dprime_ge_0p10"  : full["d_prime"] >= 0.10,
    "C_dprime_top_tert" : full["d_prime"] >= dprime_top9_thresh,
    "D_stably_aligned"  : sa_mask,
    "E_any_perm_sig"    : any_perm_mask,
}

membership_rows = []
for name, mask in subsets.items():
    for subj in full.loc[mask, "subject"]:
        membership_rows.append(dict(subset = name, subject = subj))
mem_df = pd.DataFrame(membership_rows)
mem_df.to_csv(OUT_PATH / "ch31_subset_membership.csv", index = False)

size_rows = []
for name, mask in subsets.items():
    size_rows.append(dict(subset = name, n = int(mask.sum()),
                           subjects = ",".join(sorted(full.loc[mask,
                                                                 "subject"]))))
size_df = pd.DataFrame(size_rows)
size_df.to_csv(OUT_PATH / "ch31_subset_size_summary.csv", index = False)
print("=== Subset sizes (out of N=26 in full_sentence) ===", flush = True)
print(size_df[["subset", "n"]].to_string(index = False), flush = True)


# ============================================================================
# Pairwise overlap of subsets
# ============================================================================

names = list(subsets.keys())
overlap_rows = []
for i, a in enumerate(names):
    for b in names[i:]:
        common = (subsets[a] & subsets[b]).sum()
        overlap_rows.append(dict(set_a = a, set_b = b,
                                  intersection = int(common),
                                  jaccard = float(common
                                                  / max(1, (subsets[a]
                                                             | subsets[b]).sum()))))
overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(OUT_PATH / "ch31_subset_overlap.csv", index = False)
print("\n=== Pairwise subset overlap (intersection) ===", flush = True)
print(overlap_df.pivot(index = "set_a", columns = "set_b",
                        values = "intersection").fillna("").to_string(),
      flush = True)


# ============================================================================
# (1) Within-full_sentence performance per subset
# ============================================================================

within_rows = []
for name, mask in subsets.items():
    sub = full[mask]
    if len(sub) < 4:
        continue
    for col in PERF_COLS:
        v = sub[col].dropna().values
        W, p = wilcoxon(v)
        within_rows.append(dict(subset = name, metric = col,
                                 n = len(v),
                                 mean = float(np.mean(v)),
                                 median = float(np.median(v)),
                                 W = float(W), p_two = float(p),
                                 note = ("CIRCULAR for d' if subset is "
                                          "defined by d'") if "dprime" in name
                                          and col == "d_prime"
                                          else ""))
within_df = pd.DataFrame(within_rows)
within_df.to_csv(OUT_PATH / "ch31_subset_perf_within_full.csv", index = False)
print("\n=== Within-full_sentence Wilcoxon (per subset) ===", flush = True)
print(within_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# (2) IMAGINED-block transfer (NOT circular)
# ============================================================================

trans_rows = []
for name, mask in subsets.items():
    capable_subjects = set(full.loc[mask, "subject"])
    sub_imag = imag[imag["subject"].isin(capable_subjects)]
    if len(sub_imag) < 4:
        continue
    for col in PERF_COLS:
        v = sub_imag[col].dropna().values
        W, p = wilcoxon(v)
        trans_rows.append(dict(subset = name, metric = col,
                                n = len(v),
                                mean   = float(np.mean(v)),
                                median = float(np.median(v)),
                                W = float(W), p_two = float(p)))
trans_df = pd.DataFrame(trans_rows)
trans_df.to_csv(OUT_PATH / "ch31_subset_perf_imagined_transfer.csv",
                 index = False)
print("\n=== IMAGINED-block performance for capable subsets "
      "(transfer test, NOT circular) ===", flush = True)
print(trans_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# (3) Questionnaire descriptive means and Mann-Whitney capable vs. rest
# ============================================================================

q_mean_rows = []
mwu_rows = []
for name, mask in subsets.items():
    capable = full[mask]
    rest    = full[~mask]
    for q in QUEST_COLS:
        cv = capable[q].dropna().values
        rv = rest[q].dropna().values
        if len(cv) < 4 or len(rv) < 4:
            U, p = (np.nan, np.nan)
        else:
            U, p = mannwhitneyu(cv, rv, alternative = "two-sided")
        q_mean_rows.append(dict(subset = name, quest = q,
                                  n_capable = len(cv), n_rest = len(rv),
                                  mean_capable = float(np.mean(cv)),
                                  mean_rest    = float(np.mean(rv))))
        mwu_rows.append(dict(subset = name, quest = q,
                              n_capable = len(cv), n_rest = len(rv),
                              mean_capable = float(np.mean(cv)),
                              mean_rest    = float(np.mean(rv)),
                              U = (float(U) if not np.isnan(U) else np.nan),
                              p_two = (float(p) if not np.isnan(p) else np.nan)))
qmean_df = pd.DataFrame(q_mean_rows)
qmean_df.to_csv(OUT_PATH / "ch31_subset_questionnaire_means.csv",
                 index = False)
mwu_df = pd.DataFrame(mwu_rows)
mwu_df.to_csv(OUT_PATH / "ch31_subset_questionnaire_mwu.csv", index = False)
print("\n=== Questionnaire MWU (capable vs rest) -- top 10 by p ===",
      flush = True)
print(mwu_df.sort_values("p_two").head(15).round(4)
        .to_string(index = False), flush = True)


# ============================================================================
# (4) Spearman within capable subset (BAIS focus)
# ============================================================================

within_spear = []
for name, mask in subsets.items():
    sub = full[mask]
    if len(sub) < 6:
        continue
    for q in QUEST_COLS:
        for p_col in PERF_COLS + ["accuracy", "hit_rate", "fa_rate",
                                    "criterion"]:
            x = sub[q].values; y = sub[p_col].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 6:
                continue
            rho, pv = spearmanr(x[ok], y[ok])
            within_spear.append(dict(subset = name, quest = q,
                                       perf  = p_col, n = int(ok.sum()),
                                       rho   = float(rho),
                                       p     = float(pv)))
sp_df = pd.DataFrame(within_spear)
sp_df.to_csv(OUT_PATH / "ch31_subset_spearman_within.csv", index = False)
print("\n=== Spearman within capable subset (sorted by p, top 20) ===",
      flush = True)
print(sp_df.sort_values("p").head(20).round(4)
        .to_string(index = False), flush = True)

print(f"\nChapter 31 done.  Outputs: {OUT_PATH}", flush = True)
