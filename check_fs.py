from pathlib import Path
import csv
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Load shared helpers from stimuli-analyses
sys.path.insert(0, str(Path(__file__).parent / "stimuli-analyses"))
from book3_helpers import load_block_order, load_questionnaires

QUEST_COLS = ["BAIS_V", "BAIS_C", "VHQ", "TAS", "LSHS", "DES", "FSS", "SSS_mean"]
SITE_COLORS = {"UChicago": "#9467bd", "Oberlin": "#2ca02c"}

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std

def permutation_test(a, b, n_perm=10000):
    """Permutation test for difference in means"""
    obs_diff = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    perm_diffs = []
    for _ in range(n_perm):
        perm_idx = np.random.permutation(len(pooled))
        perm_a = pooled[perm_idx[:len(a)]]
        perm_b = pooled[perm_idx[len(a):]]
        perm_diffs.append(np.mean(perm_a) - np.mean(perm_b))
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) / n_perm
    return obs_diff, p_value, np.array(perm_diffs)

def bootstrap_ci(arr, n_boot=10000, ci=95):
    """Bootstrap confidence interval for mean"""
    boot_means = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    return lower, upper, boot_means

def analyze_condition(condition_name, cur_dir):
    results_dir = cur_dir / "raw_data"
    uc_results_path = results_dir / "UChicago"
    ob_results_path = results_dir / "Oberlin"

    rows = []
    for i, experimental_site_path in enumerate([uc_results_path, ob_results_path]):
        site = "UChicago" if i == 0 else "Oberlin"
        for subject_folder_path in experimental_site_path.iterdir():
            subject_id = subject_folder_path.name
            csv_path = subject_folder_path / condition_name / f"{condition_name}_{subject_id}.csv"
            with open(csv_path, mode="r") as f:
                for row in csv.DictReader(f):
                    row["site"] = site
                    rows.append(row)

    subject_scores = {}
    for row in rows:
        sid = (row["Subject Number"], row["site"])
        correct = int(row["Subject Response"] == row["Stimulus Type"])
        if sid not in subject_scores:
            subject_scores[sid] = []
        subject_scores[sid].append(correct)

    records = []
    for (subj, site), trials in subject_scores.items():
        records.append({"subject": str(int(subj)), "site": site,
                        "accuracy": float(np.mean(trials)),
                        "block_type": condition_name})
    df = pd.DataFrame(records)

    uc_acc = df[df["site"] == "UChicago"]["accuracy"].values
    ob_acc = df[df["site"] == "Oberlin"]["accuracy"].values
    return uc_acc, ob_acc, df

def print_stats(label, arr1, arr2=None, arr1_label="", arr2_label=""):
    print(f"\n{label}")
    if arr2 is None:
        print(f"  Mean={arr1.mean():.4f}, SD={arr1.std():.4f}, Median={np.median(arr1):.4f} (n={len(arr1)})")
        t, p = stats.ttest_1samp(arr1, 0.5)
        print(f"  Parametric (t-test) vs chance (0.5): t={t:.3f}, p={p:.4f}")
        w, p_w = stats.wilcoxon(arr1 - 0.5)
        print(f"  Nonparametric (Wilcoxon) vs chance: W={w:.1f}, p={p_w:.4f}")
    else:
        print(f"  {arr1_label}: Mean={arr1.mean():.4f}, SD={arr1.std():.4f} (n={len(arr1)})")
        print(f"  {arr2_label}: Mean={arr2.mean():.4f}, SD={arr2.std():.4f} (n={len(arr2)})")
        
        t, p = stats.ttest_ind(arr1, arr2)
        d = cohens_d(arr1, arr2)
        print(f"  Parametric (t-test): t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")
        
        u, p_u = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
        print(f"  Nonparametric (Mann-Whitney U): U={u:.1f}, p={p_u:.4f}")
        
        obs_diff, p_perm, _ = permutation_test(arr1, arr2)
        print(f"  Permutation test: observed diff={obs_diff:.4f}, p={p_perm:.4f}")

def main():
    cur_dir = Path(__file__).parent.resolve()
    out_dir = cur_dir / "arwen_meeting_analyses"
    out_dir.mkdir(exist_ok=True)

    # Analyze both conditions
    uc_fs, ob_fs, df_fs = analyze_condition("full_sentence", cur_dir)
    uc_is, ob_is, df_is = analyze_condition("imagined_sentence", cur_dir)
    df_all = pd.concat([df_fs, df_is], ignore_index=True)

    # Merge in block order
    bo = load_block_order()
    bo["subject"] = bo["subject"].astype(str)
    df_all = df_all.merge(
        bo[["subject", "block_type", "block_order_index", "site"]],
        on=["subject", "block_type", "site"], how="left")

    # Merge in questionnaires
    quest = load_questionnaires()
    quest["subject"] = quest["subject"].astype(str)
    df_all = df_all.merge(quest[["subject", "site"] + QUEST_COLS],
                          on=["subject", "site"], how="left")
    
    all_data = {
        "full_sentence": {"UChicago": uc_fs, "Oberlin": ob_fs},
        "imagined_sentence": {"UChicago": uc_is, "Oberlin": ob_is}
    }

    
    print("="*70)
    print("UNIFIED ANALYSES (All Subjects Pooled)")
    print("="*70)
    
    # Unified stats
    fs_pooled = np.concatenate([uc_fs, ob_fs])
    is_pooled = np.concatenate([uc_is, ob_is])
    print_stats("Full Sentence (Unified)", fs_pooled)
    print_stats("Imagined Sentence (Unified)", is_pooled)
    print_stats("Full vs Imagined (Unified)", fs_pooled, is_pooled, "Full Sentence", "Imagined Sentence")
    
    # Within-subject correlation
    if len(uc_fs) == len(uc_is):
        r_uc, p_uc = stats.pearsonr(uc_fs, uc_is)
    if len(ob_fs) == len(ob_is):
        r_ob, p_ob = stats.pearsonr(ob_fs, ob_is)
    r_all, p_all = stats.pearsonr(fs_pooled, is_pooled)
    print(f"\nCorrelation (Full vs Imagined):")
    print(f"  UChicago: r={r_uc:.3f}, p={p_uc:.4f}")
    print(f"  Oberlin: r={r_ob:.3f}, p={p_ob:.4f}")
    print(f"  Unified: r={r_all:.3f}, p={p_all:.4f}")
    
    print("\n" + "="*70)
    print("SITE-SPECIFIC ANALYSES")
    print("="*70)
    
    # UChicago
    print("\n--- UChicago ---")
    print_stats("Full Sentence", uc_fs)
    print_stats("Imagined Sentence", uc_is)
    print_stats("Full vs Imagined", uc_fs, uc_is, "Full", "Imagined")
    
    # Oberlin
    print("\n--- Oberlin ---")
    print_stats("Full Sentence", ob_fs)
    print_stats("Imagined Sentence", ob_is)
    print_stats("Full vs Imagined", ob_fs, ob_is, "Full", "Imagined")
    
    print("\n" + "="*70)
    print("PLOTS")
    print("="*70)
    
    # ========== Figure 1: Unified Analysis ==========
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))
    fig1.suptitle("UNIFIED ANALYSIS (All Subjects Pooled)", fontsize=16, fontweight="bold")
    
    # Histograms with overlay
    ax = axes1[0, 0]
    ax.hist(fs_pooled, bins=15, alpha=0.6, label="Full Sentence", color='blue', edgecolor='black')
    ax.hist(is_pooled, bins=15, alpha=0.6, label="Imagined Sentence", color='orange', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Distribution Overlay")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Box plot comparison
    ax = axes1[0, 1]
    bp = ax.boxplot([fs_pooled, is_pooled], labels=["Full Sentence", "Imagined Sentence"], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Box Plot Comparison")
    ax.set_ylim([0.3, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Violin plot
    ax = axes1[0, 2]
    parts = ax.violinplot([fs_pooled, is_pooled], positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Full Sentence", "Imagined Sentence"])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Violin Plot")
    ax.set_ylim([0.3, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Correlation scatter
    ax = axes1[1, 0]
    ax.scatter(fs_pooled, is_pooled, alpha=0.6, s=50)
    z = np.polyfit(fs_pooled, is_pooled, 1)
    p = np.poly1d(z)
    ax.plot(fs_pooled, p(fs_pooled), "r--", linewidth=2, label=f"r={r_all:.3f}")
    ax.set_xlabel("Full Sentence Accuracy")
    ax.set_ylabel("Imagined Sentence Accuracy")
    ax.set_title("Within-Subject Correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Individual differences (scatter by condition)
    ax = axes1[1, 1]
    diffs = fs_pooled - is_pooled
    ax.scatter(range(len(diffs)), diffs, alpha=0.6, s=50)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(diffs.mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean={diffs.mean():.3f}")
    ax.set_xlabel("Subject Index")
    ax.set_ylabel("Full Sentence - Imagined Sentence")
    ax.set_title("Individual Differences")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Summary stats table
    ax = axes1[1, 2]
    ax.axis('off')
    stats_text = f"""UNIFIED SUMMARY
Full Sentence:
  Mean: {fs_pooled.mean():.4f}
  SD: {fs_pooled.std():.4f}
  
Imagined Sentence:
  Mean: {is_pooled.mean():.4f}
  SD: {is_pooled.std():.4f}
  
Between Conditions:
  t={stats.ttest_ind(fs_pooled, is_pooled)[0]:.3f}
  p={stats.ttest_ind(fs_pooled, is_pooled)[1]:.4f}
  d={cohens_d(fs_pooled, is_pooled):.3f}"""
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(out_dir / "unified_analysis.png", dpi=150, bbox_inches='tight')
    
    # ========== Figure 2: Site-Specific Analyses ==========
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
    fig2.suptitle("SITE-SPECIFIC ANALYSES", fontsize=16, fontweight="bold")
    
    for site_idx, (site_label, site_fs, site_is) in enumerate([("UChicago", uc_fs, uc_is), ("Oberlin", ob_fs, ob_is)]):
        # Box plots
        ax = axes2[0, site_idx]
        bp = ax.boxplot([site_fs, site_is], labels=["Full Sentence", "Imagined Sentence"], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue' if site_idx == 0 else 'lightcoral')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{site_label}: Box Plot")
        ax.set_ylim([0.3, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Violin plots
        ax = axes2[1, site_idx]
        parts = ax.violinplot([site_fs, site_is], positions=[1, 2], showmeans=True, showmedians=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Full Sentence", "Imagined Sentence"])
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{site_label}: Violin Plot")
        ax.set_ylim([0.3, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Correlation & stats
        ax = axes2[2, site_idx]
        r_site, p_site = stats.pearsonr(site_fs, site_is)
        ax.scatter(site_fs, site_is, alpha=0.6, s=60)
        z = np.polyfit(site_fs, site_is, 1)
        p_fit = np.poly1d(z)
        ax.plot(site_fs, p_fit(site_fs), "r--", linewidth=2)
        ax.set_xlabel("Full Sentence Accuracy")
        ax.set_ylabel("Imagined Sentence Accuracy")
        t_site, p_t = stats.ttest_ind(site_fs, site_is)
        d_site = cohens_d(site_fs, site_is)
        stats_text = f"{site_label}\nr={r_site:.3f}, p={p_site:.3f}\nt={t_site:.3f}, p={p_t:.3f}\nd={d_site:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "site_specific_analysis.png", dpi=150, bbox_inches='tight')
    
    # ========== Figure 3: Nonparametric Tests ==========
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))
    fig3.suptitle("NONPARAMETRIC & ROBUST ANALYSES", fontsize=16, fontweight="bold")
    
    # Permutation test: Full vs Imagined (unified)
    ax = axes3[0, 0]
    obs_diff_main, p_perm_main, perm_diffs_main = permutation_test(fs_pooled, is_pooled, n_perm=10000)
    ax.hist(perm_diffs_main, bins=60, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(obs_diff_main, color='red', linestyle='--', linewidth=2.5, label=f'Observed: {obs_diff_main:.4f}')
    ax.axvline(-obs_diff_main, color='red', linestyle='--', linewidth=2.5, alpha=0.5)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test: Full vs Imagined\np={p_perm_main:.4f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Bootstrap CI: Full Sentence
    ax = axes3[0, 1]
    lower_fs, upper_fs, boot_fs = bootstrap_ci(fs_pooled, n_boot=10000, ci=95)
    ax.hist(boot_fs, bins=60, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(fs_pooled.mean(), color='darkgreen', linestyle='-', linewidth=2.5, label=f'Mean: {fs_pooled.mean():.4f}')
    ax.axvline(lower_fs, color='red', linestyle='--', linewidth=2, label=f'95% CI: [{lower_fs:.4f}, {upper_fs:.4f}]')
    ax.axvline(upper_fs, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap CI: Full Sentence")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Bootstrap CI: Imagined Sentence
    ax = axes3[0, 2]
    lower_is, upper_is, boot_is = bootstrap_ci(is_pooled, n_boot=10000, ci=95)
    ax.hist(boot_is, bins=60, alpha=0.7, color='lightsalmon', edgecolor='black')
    ax.axvline(is_pooled.mean(), color='darkred', linestyle='-', linewidth=2.5, label=f'Mean: {is_pooled.mean():.4f}')
    ax.axvline(lower_is, color='blue', linestyle='--', linewidth=2, label=f'95% CI: [{lower_is:.4f}, {upper_is:.4f}]')
    ax.axvline(upper_is, color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap CI: Imagined Sentence")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Permutation test: UChicago Full vs Imagined
    ax = axes3[1, 0]
    obs_uc, p_uc_perm, perm_uc = permutation_test(uc_fs, uc_is, n_perm=10000)
    ax.hist(perm_uc, bins=60, alpha=0.7, color='mediumpurple', edgecolor='black')
    ax.axvline(obs_uc, color='red', linestyle='--', linewidth=2.5, label=f'Observed: {obs_uc:.4f}')
    ax.axvline(-obs_uc, color='red', linestyle='--', linewidth=2.5, alpha=0.5)
    ax.set_xlabel("Mean Difference (Full - Imagined)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test: UChicago\np={p_uc_perm:.4f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Permutation test: Oberlin Full vs Imagined
    ax = axes3[1, 1]
    obs_ob, p_ob_perm, perm_ob = permutation_test(ob_fs, ob_is, n_perm=10000)
    ax.hist(perm_ob, bins=60, alpha=0.7, color='wheat', edgecolor='black')
    ax.axvline(obs_ob, color='red', linestyle='--', linewidth=2.5, label=f'Observed: {obs_ob:.4f}')
    ax.axvline(-obs_ob, color='red', linestyle='--', linewidth=2.5, alpha=0.5)
    ax.set_xlabel("Mean Difference (Full - Imagined)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test: Oberlin\np={p_ob_perm:.4f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Effect size comparison with CIs
    ax = axes3[1, 2]
    comparisons = ["Full vs\nImagined\n(Unified)", "Full vs\nImagined\n(UChicago)", "Full vs\nImagined\n(Oberlin)"]
    d_vals = [cohens_d(fs_pooled, is_pooled), cohens_d(uc_fs, uc_is), cohens_d(ob_fs, ob_is)]
    colors_d = ['steelblue', 'lightblue', 'lightcoral']
    
    x_pos = np.arange(len(comparisons))
    ax.bar(x_pos, d_vals, color=colors_d, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparisons, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for i, d in enumerate(d_vals):
        ax.text(i, d + 0.01, f"{d:.3f}", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / "nonparametric_analysis.png", dpi=150, bbox_inches='tight')

    # ========== Figure 4: Block-Order Effects ==========
    rng = np.random.default_rng(0)
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle("BLOCK-ORDER EFFECTS ON ACCURACY", fontsize=14, fontweight="bold")

    for col_idx, blk in enumerate(("full_sentence", "imagined_sentence")):
        ax = axes4[col_idx]
        sub = df_all[df_all["block_type"] == blk].dropna(subset=["block_order_index"])
        for k, order in enumerate([1, 2]):
            v = sub[sub["block_order_index"] == order]["accuracy"].values
            x = np.full(len(v), k, dtype=float) + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, color=["#1f77b4", "#d62728"][k], edgecolor="k",
                       s=55, alpha=0.85, label=f"Block {order} (n={len(v)})")
            if len(v):
                ax.hlines(v.mean(), k - 0.25, k + 0.25, color="k", lw=2.5)
        # stats
        b1 = sub[sub["block_order_index"] == 1]["accuracy"].dropna().values
        b2 = sub[sub["block_order_index"] == 2]["accuracy"].dropna().values
        if len(b1) > 1 and len(b2) > 1:
            u_stat, u_p = stats.mannwhitneyu(b1, b2, alternative="two-sided")
            obs, p_perm, _ = permutation_test(b1, b2, n_perm=5000)
            ax.set_title(f"{blk}\nMW U p={u_p:.3f}  Perm p={p_perm:.3f}", fontsize=10)
        else:
            ax.set_title(blk)
        ax.axhline(0.5, color="red", lw=1.2, ls="--", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Block 1", "Block 2"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0.3, 1.0])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "block_order_analysis.png", dpi=150, bbox_inches='tight')

    # ========== Figure 5: Questionnaire Correlations (Spearman) ==========
    print("\n" + "="*70)
    print("QUESTIONNAIRE CORRELATES (Spearman rho)")
    print("="*70)

    spear_rows = []
    for blk, g in df_all.groupby("block_type"):
        for q in QUEST_COLS:
            x = g[q].values; y = g["accuracy"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 5:
                continue
            rho, pv = spearmanr(x[ok], y[ok])
            spear_rows.append(dict(block_type=blk, questionnaire=q,
                                   n=int(ok.sum()), rho=float(rho), p=float(pv)))
    spear_df = pd.DataFrame(spear_rows)
    n_tests = len(spear_df)
    spear_df["p_bonf"] = (spear_df["p"] * n_tests).clip(upper=1.0)
    print(f"Total tests: {n_tests}  (Bonf threshold: {0.05/n_tests:.4f})")
    print("\nUncorrected p<0.05:")
    print(spear_df[spear_df["p"] < 0.05].sort_values("p").round(4).to_string(index=False))
    print("\nBonferroni significant:")
    print(spear_df[spear_df["p_bonf"] < 0.05].sort_values("p").round(4).to_string(index=False))

    # Scatter plots for key questionnaires: BAIS_V, BAIS_C, LSHS, TAS
    key_quests = [q for q in ["BAIS_V", "BAIS_C", "LSHS", "TAS"] if q in df_all.columns]
    if key_quests:
        fig5, axes5 = plt.subplots(2, len(key_quests), figsize=(4 * len(key_quests), 9))
        fig5.suptitle("QUESTIONNAIRE vs ACCURACY (Spearman)", fontsize=14, fontweight="bold")
        if len(key_quests) == 1:
            axes5 = axes5.reshape(2, 1)

        for col_idx, q in enumerate(key_quests):
            for row_idx, blk in enumerate(("full_sentence", "imagined_sentence")):
                ax = axes5[row_idx, col_idx]
                g = df_all[df_all["block_type"] == blk]
                for site, gg in g.groupby("site"):
                    ax.scatter(gg[q], gg["accuracy"],
                               color=SITE_COLORS[site], edgecolor="k",
                               s=55, alpha=0.85, label=site)
                x_all = g[q].values; y_all = g["accuracy"].values
                ok = ~(np.isnan(x_all) | np.isnan(y_all))
                if ok.sum() > 3:
                    rho, pv = spearmanr(x_all[ok], y_all[ok])
                    slope, intc = np.polyfit(x_all[ok], y_all[ok], 1)
                    xs = np.linspace(x_all[ok].min(), x_all[ok].max(), 50)
                    ax.plot(xs, slope * xs + intc, "k--", lw=1.5, alpha=0.7)
                    ax.set_title(f"{blk}\n{q}  rho={rho:+.2f}  p={pv:.3f}", fontsize=9)
                else:
                    ax.set_title(f"{blk}\n{q}")
                ax.axhline(0.5, color="gray", lw=0.6, ls=":")
                ax.set_xlabel(q)
                ax.set_ylabel("Accuracy")
                ax.grid(alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / "questionnaire_correlations.png", dpi=150, bbox_inches='tight')

    # Median split on BAIS_V (the primary interest per ch29)
    print("\n" + "="*70)
    print("BAIS_V MEDIAN SPLIT (accuracy: high vs low)")
    print("="*70)
    if "BAIS_V" in df_all.columns:
        for blk, g in df_all.groupby("block_type"):
            g = g.dropna(subset=["BAIS_V"])
            cutoff = g["BAIS_V"].median()
            high = g[g["BAIS_V"] > cutoff]["accuracy"].dropna().values
            low  = g[g["BAIS_V"] <= cutoff]["accuracy"].dropna().values
            u_stat, u_p = stats.mannwhitneyu(high, low, alternative="greater")
            obs_d, p_perm, _ = permutation_test(high, low, n_perm=5000)
            print(f"\n{blk}  (cutoff BAIS_V={cutoff:.1f})")
            print(f"  High BAIS_V: mean={high.mean():.4f} (n={len(high)})  "
                  f"Low: mean={low.mean():.4f} (n={len(low)})")
            print(f"  MW one-sided (high>low): U={u_stat:.1f}, p={u_p:.4f}")
            print(f"  Permutation (obs diff={obs_d:+.4f}): p={p_perm:.4f}")

    print(f"\nPlots saved to {out_dir}/")
    print("  - unified_analysis.png")
    print("  - site_specific_analysis.png")
    print("  - nonparametric_analysis.png")
    print("  - block_order_analysis.png")
    print("  - questionnaire_correlations.png")

if __name__ == "__main__":
    main()