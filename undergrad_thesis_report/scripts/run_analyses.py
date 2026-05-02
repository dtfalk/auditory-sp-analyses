"""Undergraduate thesis report -- comprehensive analysis script.

Single script that produces every CSV and PNG used by ../report.tex.

Focus per user (David):
    1. Full sentence vs imagined sentence (the two block types).
    2. Oberlin vs UChicago site comparison.
    3. Block order (least important).
    Every analysis is reported THREE times: UChicago alone, Oberlin alone,
    Unified.  Where line-fits are involved (questionnaire scatters), each of
    the three groups gets its OWN scatter and its OWN regression line in its
    OWN panel -- never one combined line over coloured dots.

Inputs (already produced by earlier chapters):
    stimuli-analyses/outputs/book2/ch16-sdt/ch16_per_subject_sdt.csv
    stimuli-analyses/outputs/book3/ch22-classification-image/...
    stimuli-analyses/outputs/book3/ch24-spectral-logistic/...
    stimuli-analyses/outputs/book3/ch25-reminder-anchored/...
    stimuli-analyses/book3_helpers.load_questionnaires / load_block_order

Outputs:
    undergrad_thesis_report/figures/*.png
    undergrad_thesis_report/tables/*.csv
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from scipy.stats import (wilcoxon, mannwhitneyu, spearmanr,
                          ttest_1samp, ttest_ind, ttest_rel)

import sys
PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "stimuli-analyses"))
from book3_helpers import load_questionnaires, load_block_order

REPORT_ROOT = Path(__file__).resolve().parents[1]
FIG_PATH    = REPORT_ROOT / "figures"
TBL_PATH    = REPORT_ROOT / "tables"
FIG_PATH.mkdir(parents = True, exist_ok = True)
TBL_PATH.mkdir(parents = True, exist_ok = True)

ANALY_PATH = PROJ / "stimuli-analyses" / "outputs"

QUEST_COLS = ["BAIS_V", "BAIS_C", "VHQ", "TAS", "LSHS", "DES", "FSS",
              "SSS_mean"]
SITE_COLOR = {"UChicago": "#9467bd", "Oberlin": "#2ca02c",
               "Unified":  "#1f77b4"}
BLOCK_LABEL = {"full_sentence":     "Full Sentence",
                "imagined_sentence": "Imagined Sentence"}

RNG = np.random.default_rng(20260501)


# ============================================================================
# Loading and merging per-subject data
# ============================================================================

def load_full_table():
    """Return one row per (subject, block_type) with site, all SDT columns,
    Ch.22/24/25 metrics, questionnaires, and block_order_index."""
    sdt = pd.read_csv(ANALY_PATH / "book2" / "ch16-sdt"
                       / "ch16_per_subject_sdt.csv")
    ci  = pd.read_csv(ANALY_PATH / "book3" / "ch22-classification-image"
                       / "ch22_per_subject_ci.csv")
    sp  = pd.read_csv(ANALY_PATH / "book3" / "ch24-spectral-logistic"
                       / "ch24_per_subject_spectral.csv")
    sl  = pd.read_csv(ANALY_PATH / "book3" / "ch25-reminder-anchored"
                       / "ch25_per_subject_slopes.csv")
    for d in (sdt, ci, sp, sl):
        d["subject"] = d["subject"].astype(str)

    out = (sdt[["site", "subject", "block_type", "hit_rate", "fa_rate",
                  "d_prime", "criterion", "accuracy", "p_perm_dprime"]]
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
    out = out.merge(quest[["subject"] + QUEST_COLS], on = "subject",
                     how = "left")
    bo = load_block_order()
    bo["subject"] = bo["subject"].astype(str)
    out = out.merge(bo[["site", "subject", "block_type",
                          "block_order_index"]],
                     on = ["site", "subject", "block_type"], how = "left")
    return out


df = load_full_table()
print(f"Loaded {len(df)} rows ({df['subject'].nunique()} subjects, "
      f"sites: {dict(df.groupby('site')['subject'].nunique())})",
      flush = True)


# ============================================================================
# Helpers
# ============================================================================

def cohens_d_one_sample(x, mu = 0.0):
    x = np.asarray(x, dtype = float)
    return (np.mean(x) - mu) / np.std(x, ddof = 1)


def cohens_d_two_sample(a, b):
    a = np.asarray(a, dtype = float); b = np.asarray(b, dtype = float)
    s = np.sqrt(((len(a) - 1) * np.var(a, ddof = 1)
                  + (len(b) - 1) * np.var(b, ddof = 1))
                  / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / s


def boot_ci_mean(x, n_boot = 10000, ci = 95, rng = RNG):
    x = np.asarray(x, dtype = float)
    boots = rng.choice(x, size = (n_boot, len(x)), replace = True).mean(axis = 1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(np.mean(x)), float(lo), float(hi)


def perm_test_diff(a, b, n_perm = 10000, rng = RNG, paired = False):
    """Two-sided permutation test on mean difference."""
    a = np.asarray(a, dtype = float); b = np.asarray(b, dtype = float)
    if paired:
        d = a - b
        obs = float(np.mean(d))
        signs = rng.choice([-1, 1], size = (n_perm, len(d)))
        perm = (signs * d).mean(axis = 1)
    else:
        obs   = float(np.mean(a) - np.mean(b))
        pool  = np.concatenate([a, b])
        nA    = len(a)
        perm  = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(pool)
            perm[i] = pool[:nA].mean() - pool[nA:].mean()
    p = float((np.abs(perm) >= np.abs(obs) - 1e-12).mean())
    return obs, p


def split_groups(d, block):
    sub = d[d["block_type"] == block]
    return {"UChicago": sub[sub["site"] == "UChicago"],
             "Oberlin":  sub[sub["site"] == "Oberlin"],
             "Unified":  sub}


# ============================================================================
# SECTION A.  Accuracy descriptives + tests vs. chance (3-way x 2 blocks)
# ============================================================================

print("\n=== Section A: accuracy vs chance (per group, per block) ===",
      flush = True)
rows = []
for block in ("full_sentence", "imagined_sentence"):
    g = split_groups(df, block)
    for name, sub in g.items():
        x = sub["accuracy"].values
        mean, lo, hi = boot_ci_mean(x)
        # one-sample Wilcoxon vs 0.5
        try:
            W, p_w = wilcoxon(x - 0.5)
        except ValueError:
            W, p_w = (np.nan, np.nan)
        t, p_t = ttest_1samp(x, 0.5)
        d = cohens_d_one_sample(x, 0.5)
        rows.append(dict(block = block, group = name, n = len(x),
                          mean = mean, sd = float(np.std(x, ddof = 1)),
                          ci_lo = lo, ci_hi = hi,
                          wilcoxon_W = float(W), wilcoxon_p = float(p_w),
                          t = float(t), t_p = float(p_t),
                          cohens_d = float(d)))
acc_vs_chance = pd.DataFrame(rows)
acc_vs_chance.to_csv(TBL_PATH / "A_accuracy_vs_chance.csv", index = False)
print(acc_vs_chance.round(4).to_string(index = False), flush = True)


# Plot: accuracy distribution per group per block, with chance line
fig, axes = plt.subplots(2, 3, figsize = (12, 6.5), sharey = True)
for r, block in enumerate(("full_sentence", "imagined_sentence")):
    g = split_groups(df, block)
    for c, name in enumerate(("UChicago", "Oberlin", "Unified")):
        ax  = axes[r, c]
        x   = g[name]["accuracy"].values
        ax.hist(x, bins = 12, color = SITE_COLOR[name], alpha = 0.75,
                 edgecolor = "k")
        ax.axvline(0.5, color = "red", linestyle = "--", linewidth = 1.5)
        ax.axvline(np.mean(x), color = "black", linestyle = "-",
                    linewidth = 2)
        m, lo, hi = boot_ci_mean(x)
        row = acc_vs_chance.query("block==@block and group==@name").iloc[0]
        ax.set_title(f"{name} -- {BLOCK_LABEL[block]}\n"
                      f"mean = {m:.3f}  [{lo:.3f}, {hi:.3f}]   "
                      f"Wilcoxon p = {row['wilcoxon_p']:.3f}",
                      fontsize = 9)
        ax.set_xlabel("accuracy")
        if c == 0:
            ax.set_ylabel("# subjects")
        ax.set_xlim(0.40, 0.60)
plt.suptitle("Section A.  Accuracy vs chance, three-way split per block",
              fontsize = 12)
plt.tight_layout()
plt.savefig(FIG_PATH / "A_accuracy_vs_chance.png", dpi = 140)
plt.close()


# ============================================================================
# SECTION B.  Full vs Imagined within each group (paired)
# ============================================================================

print("\n=== Section B: full vs imagined (paired, 3-way) ===", flush = True)
b_rows = []
for name in ("UChicago", "Oberlin", "Unified"):
    if name == "Unified":
        sub = df
    else:
        sub = df[df["site"] == name]
    wide = sub.pivot(index = "subject", columns = "block_type",
                      values = "accuracy").dropna()
    fs = wide["full_sentence"].values
    im = wide["imagined_sentence"].values
    diffs = fs - im
    W, p_w = wilcoxon(diffs)
    t, p_t = ttest_rel(fs, im)
    obs, p_perm = perm_test_diff(fs, im, paired = True)
    d = cohens_d_one_sample(diffs, 0.0)
    b_rows.append(dict(group = name, n = len(diffs),
                        mean_full = float(np.mean(fs)),
                        mean_imag = float(np.mean(im)),
                        mean_diff = float(np.mean(diffs)),
                        sd_diff = float(np.std(diffs, ddof = 1)),
                        wilcoxon_W = float(W), wilcoxon_p = float(p_w),
                        paired_t = float(t), t_p = float(p_t),
                        perm_p = float(p_perm),
                        cohens_d_paired = float(d)))
fs_vs_im = pd.DataFrame(b_rows)
fs_vs_im.to_csv(TBL_PATH / "B_full_vs_imagined.csv", index = False)
print(fs_vs_im.round(4).to_string(index = False), flush = True)


fig, axes = plt.subplots(1, 3, figsize = (12, 4.5))
for ax, name in zip(axes, ("UChicago", "Oberlin", "Unified")):
    if name == "Unified":
        sub = df
    else:
        sub = df[df["site"] == name]
    wide = sub.pivot(index = "subject", columns = "block_type",
                      values = "accuracy").dropna()
    for _, r in wide.iterrows():
        ax.plot([0, 1], [r["full_sentence"], r["imagined_sentence"]],
                 color = "gray", alpha = 0.5, marker = "o")
    ax.plot([0, 1], [wide["full_sentence"].mean(),
                       wide["imagined_sentence"].mean()],
             color = SITE_COLOR[name], linewidth = 3, marker = "s",
             markersize = 10, label = "group mean")
    ax.axhline(0.5, color = "red", linestyle = "--", alpha = 0.7)
    row = fs_vs_im.query("group==@name").iloc[0]
    ax.set_title(f"{name}  (n = {int(row['n'])})\n"
                  f"diff = {row['mean_diff']:+.3f}, "
                  f"Wilcoxon p = {row['wilcoxon_p']:.3f}, "
                  f"d = {row['cohens_d_paired']:+.2f}",
                  fontsize = 10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full", "Imagined"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.40, 0.60)
plt.suptitle("Section B.  Within-subject Full vs Imagined accuracy",
              fontsize = 12)
plt.tight_layout()
plt.savefig(FIG_PATH / "B_full_vs_imagined.png", dpi = 140)
plt.close()


# ============================================================================
# SECTION C.  Site comparison Oberlin vs UChicago, per block (unpaired)
# ============================================================================

print("\n=== Section C: Oberlin vs UChicago per block ===", flush = True)
c_rows = []
for block in ("full_sentence", "imagined_sentence"):
    sub = df[df["block_type"] == block]
    a = sub[sub["site"] == "Oberlin"]["accuracy"].values
    b = sub[sub["site"] == "UChicago"]["accuracy"].values
    U, p_u = mannwhitneyu(a, b, alternative = "two-sided")
    t, p_t = ttest_ind(a, b, equal_var = False)
    obs, p_perm = perm_test_diff(a, b, paired = False)
    d = cohens_d_two_sample(a, b)
    c_rows.append(dict(block = block,
                        n_oberlin = len(a), n_uchicago = len(b),
                        mean_oberlin = float(np.mean(a)),
                        mean_uchicago = float(np.mean(b)),
                        diff = float(np.mean(a) - np.mean(b)),
                        mwu_U = float(U), mwu_p = float(p_u),
                        welch_t = float(t), welch_p = float(p_t),
                        perm_p = float(p_perm),
                        cohens_d = float(d)))
site_df = pd.DataFrame(c_rows)
site_df.to_csv(TBL_PATH / "C_site_comparison.csv", index = False)
print(site_df.round(4).to_string(index = False), flush = True)


fig, axes = plt.subplots(1, 2, figsize = (10, 5))
for ax, block in zip(axes, ("full_sentence", "imagined_sentence")):
    sub = df[df["block_type"] == block]
    data, colors, labels = [], [], []
    for s in ("UChicago", "Oberlin"):
        x = sub[sub["site"] == s]["accuracy"].values
        data.append(x); colors.append(SITE_COLOR[s])
        labels.append(f"{s}\n(n = {len(x)})")
    bp = ax.boxplot(data, labels = labels, patch_artist = True,
                     widths = 0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, x in enumerate(data):
        ax.scatter(np.full_like(x, i + 1, dtype = float)
                    + RNG.uniform(-0.06, 0.06, size = len(x)),
                    x, color = colors[i], edgecolor = "k", zorder = 3)
    ax.axhline(0.5, color = "red", linestyle = "--", alpha = 0.7)
    row = site_df.query("block==@block").iloc[0]
    ax.set_title(f"{BLOCK_LABEL[block]}\n"
                  f"diff = {row['diff']:+.3f},  MW p = {row['mwu_p']:.3f},  "
                  f"d = {row['cohens_d']:+.2f}",
                  fontsize = 10)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.40, 0.60)
plt.suptitle("Section C.  Site comparison per block (unpaired)",
              fontsize = 12)
plt.tight_layout()
plt.savefig(FIG_PATH / "C_site_comparison.png", dpi = 140)
plt.close()


# ============================================================================
# SECTION D.  Block-order (briefly)
# ============================================================================

print("\n=== Section D: block order ===", flush = True)
d_rows = []
for block in ("full_sentence", "imagined_sentence"):
    g = split_groups(df, block)
    for name, sub in g.items():
        b1 = sub[sub["block_order_index"] == 1]["accuracy"].dropna().values
        b2 = sub[sub["block_order_index"] == 2]["accuracy"].dropna().values
        if len(b1) < 3 or len(b2) < 3:
            d_rows.append(dict(block = block, group = name,
                                n_b1 = len(b1), n_b2 = len(b2),
                                mean_b1 = (float(np.mean(b1))
                                            if len(b1) else np.nan),
                                mean_b2 = (float(np.mean(b2))
                                            if len(b2) else np.nan),
                                mwu_p = np.nan, cohens_d = np.nan))
            continue
        U, p_u = mannwhitneyu(b1, b2, alternative = "two-sided")
        d_rows.append(dict(block = block, group = name,
                            n_b1 = len(b1), n_b2 = len(b2),
                            mean_b1 = float(np.mean(b1)),
                            mean_b2 = float(np.mean(b2)),
                            mwu_p = float(p_u),
                            cohens_d = float(cohens_d_two_sample(b2, b1))))
order_df = pd.DataFrame(d_rows)
order_df.to_csv(TBL_PATH / "D_block_order.csv", index = False)
print(order_df.round(4).to_string(index = False), flush = True)


# ============================================================================
# SECTION E.  Convergent metrics for Oberlin full_sentence
#             (the headline finding -- multiple mathematically distinct
#              quantities all point the same way)
# ============================================================================

print("\n=== Section E: convergent metrics Oberlin full_sentence ===",
      flush = True)
metrics = [("d_prime",         "d'  (SDT sensitivity, Ch.16)",          0.0),
            ("accuracy",        "accuracy",                               0.5),
            ("cos_ci_true",     "cos(CI_subject, CI_true) (Ch.22)",       0.0),
            ("cos_w_true",      "cos(w_subject, w_true) (Ch.24)",         0.0),
            ("beta_truetarget", "beta_truetarget (per-subject logistic, "
                                "Ch.25)",                                  0.0)]

e_rows = []
for name in ("UChicago", "Oberlin", "Unified"):
    sub = df[df["block_type"] == "full_sentence"]
    if name != "Unified":
        sub = sub[sub["site"] == name]
    for col, label, mu in metrics:
        x = sub[col].dropna().values
        try:
            W, p_w = wilcoxon(x - mu)
        except ValueError:
            W, p_w = (np.nan, np.nan)
        t, p_t = ttest_1samp(x, mu)
        d = cohens_d_one_sample(x, mu)
        m, lo, hi = boot_ci_mean(x)
        e_rows.append(dict(group = name, metric = col, label = label,
                            n = len(x), null = mu,
                            mean = m, ci_lo = lo, ci_hi = hi,
                            wilcoxon_W = float(W) if not np.isnan(W) else np.nan,
                            wilcoxon_p = float(p_w) if not np.isnan(p_w) else np.nan,
                            t = float(t), t_p = float(p_t),
                            cohens_d = float(d)))
conv_df = pd.DataFrame(e_rows)
conv_df.to_csv(TBL_PATH / "E_convergent_full_sentence.csv", index = False)
print(conv_df.round(4)[["group", "metric", "n", "mean", "ci_lo",
                          "ci_hi", "wilcoxon_p", "cohens_d"]]
        .to_string(index = False), flush = True)


fig, axes = plt.subplots(len(metrics), 3, figsize = (12, 2.6 * len(metrics)),
                           sharey = "row")
for r, (col, label, mu) in enumerate(metrics):
    for c, name in enumerate(("UChicago", "Oberlin", "Unified")):
        ax  = axes[r, c]
        sub = df[df["block_type"] == "full_sentence"]
        if name != "Unified":
            sub = sub[sub["site"] == name]
        x = sub[col].dropna().values
        ax.hist(x, bins = 10, color = SITE_COLOR[name], alpha = 0.75,
                 edgecolor = "k")
        ax.axvline(mu, color = "red", linestyle = "--", linewidth = 1.5)
        ax.axvline(np.mean(x), color = "black", linestyle = "-",
                    linewidth = 2)
        row = conv_df.query("group==@name and metric==@col").iloc[0]
        ax.set_title(f"{name}: {label}\n"
                      f"mean = {row['mean']:+.3f}, "
                      f"Wilcoxon p = {row['wilcoxon_p']:.3f}, "
                      f"d = {row['cohens_d']:+.2f}", fontsize = 8)
        if r == len(metrics) - 1:
            ax.set_xlabel("metric value")
plt.suptitle("Section E.  Four mathematically distinct alignment statistics, "
              "full_sentence block, three-way split", fontsize = 12)
plt.tight_layout()
plt.savefig(FIG_PATH / "E_convergent_full_sentence.png", dpi = 140)
plt.close()


# ============================================================================
# SECTION F.  Questionnaire correlations
#   -- per block, separate UChicago / Oberlin / Unified scatter+line panel
#   -- primary outcome variable: accuracy
# ============================================================================

print("\n=== Section F: questionnaire vs accuracy ===", flush = True)
f_rows = []
for block in ("full_sentence", "imagined_sentence"):
    g = split_groups(df, block)
    for q in QUEST_COLS:
        for name, sub in g.items():
            x = sub[q].values
            y = sub["accuracy"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 4:
                continue
            rho, p = spearmanr(x[ok], y[ok])
            f_rows.append(dict(block = block, group = name, quest = q,
                                n = int(ok.sum()),
                                rho = float(rho), p = float(p)))
quest_df = pd.DataFrame(f_rows)
quest_df.to_csv(TBL_PATH / "F_questionnaire_spearman.csv", index = False)
print(quest_df.sort_values("p").head(20).round(4).to_string(index = False),
       flush = True)


for block in ("full_sentence", "imagined_sentence"):
    fig, axes = plt.subplots(len(QUEST_COLS), 3,
                               figsize = (11, 2.4 * len(QUEST_COLS)))
    g = split_groups(df, block)
    for r, q in enumerate(QUEST_COLS):
        for c, name in enumerate(("UChicago", "Oberlin", "Unified")):
            ax  = axes[r, c]
            sub = g[name]
            x = sub[q].values; y = sub["accuracy"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            xx, yy = x[ok], y[ok]
            ax.scatter(xx, yy, color = SITE_COLOR[name],
                        edgecolor = "k", alpha = 0.85)
            if len(xx) >= 3 and np.ptp(xx) > 0:
                m, b = np.polyfit(xx, yy, 1)
                xs = np.array([xx.min(), xx.max()])
                ax.plot(xs, m * xs + b, color = SITE_COLOR[name],
                         linestyle = "--", linewidth = 2)
            rho, p = (spearmanr(xx, yy) if len(xx) >= 4
                       else (np.nan, np.nan))
            ax.axhline(0.5, color = "red", linestyle = ":", alpha = 0.5)
            ax.set_title(f"{name}: {q}\n"
                          f"n = {len(xx)}, rho = {rho:+.2f}, p = {p:.3f}",
                          fontsize = 8)
            if c == 0:
                ax.set_ylabel("accuracy")
            if r == len(QUEST_COLS) - 1:
                ax.set_xlabel(q)
    plt.suptitle(f"Section F.  Questionnaires vs accuracy, "
                  f"{BLOCK_LABEL[block]}.  "
                  f"Each panel has its own group, scatter and OLS line.",
                  fontsize = 11)
    plt.tight_layout()
    plt.savefig(FIG_PATH / f"F_quest_vs_accuracy_{block}.png", dpi = 130)
    plt.close()


# ============================================================================
# SECTION G.  N-sensitivity / power heuristics for the trending findings
#
#   For Oberlin full_sentence accuracy vs chance (the headline) and
#   for site Oberlin vs UChicago (the trend), produce subsample-size
#   curves: at each n_sub from 5 to N_full, draw 2000 random subsamples,
#   compute the test statistic, and report the fraction with p<0.05.
#   This shows whether the existing N is the limit, or whether the effect
#   is stable.
# ============================================================================

print("\n=== Section G: N-sensitivity / subsample power curves ===",
      flush = True)


def subsample_power_one_sample(x, mu = 0.5, n_sub_grid = None,
                                 n_boot = 2000, alpha = 0.05, rng = RNG):
    if n_sub_grid is None:
        n_sub_grid = list(range(5, len(x) + 1))
    rows = []
    for n_sub in n_sub_grid:
        sigs = 0
        means = []
        for _ in range(n_boot):
            sample = rng.choice(x, size = n_sub, replace = False)
            try:
                _, p = wilcoxon(sample - mu)
            except ValueError:
                continue
            means.append(np.mean(sample))
            if p < alpha:
                sigs += 1
        rows.append(dict(n_sub = n_sub,
                          power = sigs / n_boot,
                          mean_of_means = float(np.mean(means)),
                          sd_of_means = float(np.std(means, ddof = 1))))
    return pd.DataFrame(rows)


def subsample_power_two_sample(a, b, n_sub_grid_a = None,
                                 n_sub_grid_b = None, n_boot = 2000,
                                 alpha = 0.05, rng = RNG):
    """For each (n_a, n_b) on the same fraction of the original groups,
    estimate fraction of MWU two-sided p < alpha."""
    if n_sub_grid_a is None:
        n_sub_grid_a = list(range(5, len(a) + 1))
    if n_sub_grid_b is None:
        n_sub_grid_b = list(range(5, len(b) + 1))
    # Use proportional subsampling: keep ratio
    base_ratio = len(a) / (len(a) + len(b))
    rows = []
    for n_sub in range(10, len(a) + len(b) + 1):
        nA = max(3, int(round(n_sub * base_ratio)))
        nB = max(3, n_sub - nA)
        if nA > len(a) or nB > len(b):
            continue
        sigs = 0
        diffs = []
        for _ in range(n_boot):
            sa = rng.choice(a, size = nA, replace = False)
            sb = rng.choice(b, size = nB, replace = False)
            U, p = mannwhitneyu(sa, sb, alternative = "two-sided")
            diffs.append(np.mean(sa) - np.mean(sb))
            if p < alpha:
                sigs += 1
        rows.append(dict(n_total = n_sub, n_a = nA, n_b = nB,
                          power = sigs / n_boot,
                          mean_diff = float(np.mean(diffs)),
                          sd_diff = float(np.std(diffs, ddof = 1))))
    return pd.DataFrame(rows)


# G1: Oberlin full_sentence accuracy vs chance
ob_fs = df.query("site=='Oberlin' and block_type=='full_sentence'")[
            "accuracy"].values
g1 = subsample_power_one_sample(ob_fs, mu = 0.5)
g1.to_csv(TBL_PATH / "G_power_oberlin_fs_vs_chance.csv", index = False)

# G2: Site comparison full_sentence (Oberlin vs UChicago)
a = df.query("site=='Oberlin' and block_type=='full_sentence'")[
        "accuracy"].values
b = df.query("site=='UChicago' and block_type=='full_sentence'")[
        "accuracy"].values
g2 = subsample_power_two_sample(a, b)
g2.to_csv(TBL_PATH / "G_power_site_full_sentence.csv", index = False)

# G3: Unified full vs imagined paired (within-subject)
wide = df.pivot(index = "subject", columns = "block_type",
                  values = "accuracy").dropna()
diffs_unified = (wide["full_sentence"] - wide["imagined_sentence"]).values

def subsample_power_paired(d, n_sub_grid = None, n_boot = 2000,
                            alpha = 0.05, rng = RNG):
    if n_sub_grid is None:
        n_sub_grid = list(range(5, len(d) + 1))
    rows = []
    for n_sub in n_sub_grid:
        sigs = 0
        means = []
        for _ in range(n_boot):
            sample = rng.choice(d, size = n_sub, replace = False)
            try:
                _, p = wilcoxon(sample)
            except ValueError:
                continue
            means.append(np.mean(sample))
            if p < alpha:
                sigs += 1
        rows.append(dict(n_sub = n_sub,
                          power = sigs / n_boot,
                          mean_diff = float(np.mean(means))))
    return pd.DataFrame(rows)

g3 = subsample_power_paired(diffs_unified)
g3.to_csv(TBL_PATH / "G_power_full_vs_imagined_unified.csv", index = False)


# Plot G
fig, axes = plt.subplots(1, 3, figsize = (13, 4.2))
ax = axes[0]
ax.plot(g1["n_sub"], g1["power"], marker = "o", color = SITE_COLOR["Oberlin"])
ax.axhline(0.80, color = "red", linestyle = "--", alpha = 0.7,
            label = "0.80 power")
ax.axhline(0.05, color = "gray", linestyle = ":", alpha = 0.7,
            label = "alpha = 0.05")
ax.set_xlabel("subsample size n"); ax.set_ylabel("Pr(Wilcoxon p < 0.05)")
ax.set_title("G1.  Oberlin full_sentence accuracy vs chance\n"
              "(subsampling from observed N = " f"{len(ob_fs)})", fontsize = 10)
ax.legend(loc = "lower right", fontsize = 8); ax.set_ylim(0, 1)

ax = axes[1]
ax.plot(g2["n_total"], g2["power"], marker = "o", color = "purple")
ax.axhline(0.80, color = "red", linestyle = "--", alpha = 0.7)
ax.axhline(0.05, color = "gray", linestyle = ":", alpha = 0.7)
ax.set_xlabel("total subsample size (Oberlin + UChicago)")
ax.set_ylabel("Pr(MWU p < 0.05)")
ax.set_title("G2.  Site difference, full_sentence accuracy\n"
              f"(observed nA = {len(a)}, nB = {len(b)})", fontsize = 10)
ax.set_ylim(0, 1)

ax = axes[2]
ax.plot(g3["n_sub"], g3["power"], marker = "o",
         color = SITE_COLOR["Unified"])
ax.axhline(0.80, color = "red", linestyle = "--", alpha = 0.7)
ax.axhline(0.05, color = "gray", linestyle = ":", alpha = 0.7)
ax.set_xlabel("paired subsample size n")
ax.set_ylabel("Pr(paired Wilcoxon p < 0.05)")
ax.set_title("G3.  Full vs Imagined within-subject\n"
              f"(observed N = {len(diffs_unified)})", fontsize = 10)
ax.set_ylim(0, 1)

plt.suptitle("Section G.  Sub-sample 'is it a power problem?' diagnostics",
              fontsize = 12)
plt.tight_layout()
plt.savefig(FIG_PATH / "G_power_curves.png", dpi = 140)
plt.close()


# Estimate N required to hit 0.80 power if effect is held fixed:
def n_for_target_power(power_df, key = "n_sub", target = 0.80):
    ok = power_df[power_df["power"] >= target]
    if len(ok) == 0:
        return None
    return int(ok.iloc[0][key])

required_rows = [
    dict(comparison = "Oberlin full_sentence vs chance",
         observed_n = len(ob_fs),
         observed_p = float(acc_vs_chance.query(
            "block=='full_sentence' and group=='Oberlin'"
         ).iloc[0]["wilcoxon_p"]),
         observed_d = float(acc_vs_chance.query(
            "block=='full_sentence' and group=='Oberlin'"
         ).iloc[0]["cohens_d"]),
         n_for_80pct_power = n_for_target_power(g1, "n_sub")),
    dict(comparison = "Site Oberlin vs UChicago, full_sentence",
         observed_n = len(a) + len(b),
         observed_p = float(site_df.query(
            "block=='full_sentence'").iloc[0]["mwu_p"]),
         observed_d = float(site_df.query(
            "block=='full_sentence'").iloc[0]["cohens_d"]),
         n_for_80pct_power = n_for_target_power(g2, "n_total")),
    dict(comparison = "Full vs Imagined paired (Unified)",
         observed_n = len(diffs_unified),
         observed_p = float(fs_vs_im.query(
            "group=='Unified'").iloc[0]["wilcoxon_p"]),
         observed_d = float(fs_vs_im.query(
            "group=='Unified'").iloc[0]["cohens_d_paired"]),
         n_for_80pct_power = n_for_target_power(g3, "n_sub"))]
required_df = pd.DataFrame(required_rows)
required_df.to_csv(TBL_PATH / "G_required_n_summary.csv", index = False)
print("\n--- N required for 80% power (interpolated from subsample curves) ---",
       flush = True)
print(required_df.to_string(index = False), flush = True)


print(f"\nDone.  Tables -> {TBL_PATH}\n        Figures -> {FIG_PATH}",
       flush = True)
