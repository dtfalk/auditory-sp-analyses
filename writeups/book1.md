# Book 1. Stimulus-Only Analyses
*A structured plan for analyzing the stimulus set independently of subject responses*

## Purpose of Book 1

Book 1 is about the stimuli themselves, before bringing in any behavioral data. The central goal is to answer:

1. **What information was actually present in the stimulus set?**
2. **How much structure existed in the stimuli independent of our labels?**
3. **How much did our target/distractor split capture a real dimension of structure versus impose one?**
4. **How do these answers depend on the similarity metric used?**

This book should be organized so that each analysis is easy to interpret and clearly falls into one of two modes:

- **Label-aware analyses**  
  These know which stimuli were called targets and which were called distractors.

- **Label-blind analyses**  
  These ignore the target/distractor split and ask what structure exists in the stimulus set on its own.

A second cross-cutting organization should distinguish:

- **Pearson-based analyses**  
  These are essential because Pearson was used to define the stimuli.

- **Non-Pearson analyses**  
  These are important because subjects may have been sensitive to structure not well captured by Pearson.

---

# Global Structure of Book 1

## Part I. Foundations
Basic definitions, preprocessing, and setup used by all later chapters.

## Part II. Pearson-Centered Analyses
A thorough treatment of the stimulus set from the perspective of the metric actually used to select stimuli.

## Part III. Non-Pearson Metric Analyses
Parallel analyses using alternative similarity measures that may better reflect perceptual or structural properties.

## Part IV. Synthesis
Direct comparisons across metrics and summary conclusions about what information the stimuli contained.

---

# Part I. Foundations

## Chapter 1. Stimulus Inventory and Provenance

### Goal
Create a clean record of what the stimulus set actually is.

### Questions
- How many total stimuli were there?
- How many targets and distractors?
- What exact selection rules created them?
- What preprocessing was applied before correlation was computed?
- Were all clips equal length and identically aligned?

### Outputs
- Table of stimulus counts
- Clear written definition of:
  - template clip
  - candidate white-noise clips
  - target selection rule
  - distractor selection rule
- Diagram of workflow from raw noise pool to final stimulus set

### Notes
This chapter is descriptive, but it is crucial. Later analyses will be hard to interpret unless the exact pipeline is documented.

---

## Chapter 2. Preprocessing and Signal Standardization

### Goal
Establish a single preprocessing scheme for all analyses.

### Questions
- Were signals z-scored before Pearson was computed?
- Were amplitudes normalized?
- Were signals mean-centered?
- Were there onset offsets or trimming choices that matter?
- Was sampling rate identical across all files?

### Outputs
- One canonical preprocessing pipeline
- One rationale for why this pipeline is used for all Book 1 analyses
- One sensitivity check showing whether conclusions depend on preprocessing choice

### Notes
This chapter protects against downstream confusion where differences are actually artifacts of scaling, alignment, or normalization.

---

## Chapter 3. Core Descriptive Properties of the Stimulus Set

### Goal
Characterize the basic physical properties of the stimuli before any similarity analysis.

### Questions
- What are the amplitude distributions?
- Do targets and distractors differ in RMS energy, peak amplitude, variance, or envelope shape?
- Are there obvious outliers?

### Outputs
- Histograms of energy and amplitude measures
- Summary statistics by group
- Outlier report

### Label status
Label-aware

### Metric status
Metric-independent

### Notes
This chapter checks whether there are low-level differences between groups that are unrelated to the intended manipulation.

---

# Part II. Pearson-Centered Analyses

## Why Part II matters

Pearson defined the target/distractor split. So even if Pearson is not perceptually ideal, it must be treated as a primary analytic framework. This part should answer:

- What does the stimulus set look like from the perspective of Pearson?
- What did the Pearson-based selection actually produce?
- How much structure exists under the same metric used to define the categories?

---

## Chapter 4. Distribution of Template Correlations

### Goal
Understand the distribution of Pearson correlations between all candidate noise clips and the "wall" template.

### Questions
- What does the full distribution of Pearson values look like across the original 2 million clips?
- Where do the selected targets fall in that distribution?
- Where do the selected distractors fall?
- How extreme were the selected targets?
- How close to zero were the selected distractors?

### Outputs
- Histogram or density plot of correlation values for the full candidate pool
- Overlaid markers for selected targets and distractors
- Quantiles and tail probabilities
- Summary of separation between selected groups

### Label status
Label-aware

### Metric status
Pearson

### Notes
This chapter establishes how selective the original selection procedure actually was.

---

## Chapter 5. Pairwise Pearson Structure Within and Across Groups

### Goal
Characterize pairwise similarity among stimuli using Pearson.

### Questions
- What is the distribution of pairwise Pearson values among targets?
- What is the distribution among distractors?
- What is the distribution between targets and distractors?
- Are targets more mutually similar than distractors?
- Are within-group and between-group distributions separable?

### Outputs
- Three distributions:
  - target-target
  - distractor-distractor
  - target-distractor
- Mean, median, variance for each
- Effect sizes comparing distributions
- Correlation matrix heatmap ordered by group

### Label status
Label-aware

### Metric status
Pearson

### Notes
This is one of the core chapters. It tells you what the selected groups actually look like internally under the same metric that defined them.

---

## Chapter 6. Target Cohesion as a Function of Selection Rank

### Goal
Understand whether higher-ranked targets are also more mutually similar.

### Questions
- If targets are ordered by correlation with the template, does within-target similarity increase with rank?
- Is the top 20 much more cohesive than the top 150?
- Does pairwise cohesion rise smoothly or only in the extreme tail?

### Outputs
- Curves showing average within-group Pearson for:
  - top 10
  - top 20
  - top 50
  - top 100
  - top 150
- Rank versus mean similarity plots
- Stability estimates across subset sizes

### Label status
Label-aware

### Metric status
Pearson

### Notes
This chapter addresses the question of whether selecting by Pearson to the template also induces increasing internal structure.

---

## Chapter 7. Relationship Between Template Alignment and Pairwise Cohesion

### Goal
Test whether stronger template similarity predicts stronger similarity to other targets.

### Questions
- For each target, does higher correlation with "wall" predict higher average correlation with the rest of the target set?
- Is the target set dominated by a shared component or by idiosyncratic noise?

### Outputs
- Scatterplot:
  - x-axis = Pearson to template
  - y-axis = mean Pearson to other targets
- Correlation and regression summary
- Residual analysis to identify unusual stimuli

### Label status
Label-aware

### Metric status
Pearson

### Notes
This chapter is conceptually important because it tests whether the target set forms a coherent family or just a list of individually extreme clips.
0

---

## Chapter 8. Pearson-Based Geometry of the Full Stimulus Set

### Goal
Map the shape of the stimulus set using pairwise Pearson similarities, including label-blind analyses.

### Questions
- Do the stimuli naturally cluster under Pearson?
- Do those natural clusters align with target/distractor labels?
- Is the target/distractor split a dominant axis of organization?

### Outputs
- Similarity matrix and corresponding distance matrix
- MDS or PCA based on Pearson-derived distances
- Clustering solutions with k = 2 and possibly larger k
- Visual comparison of intrinsic clusters versus experimenter labels

### Label status
Mixed
- First label-blind
- Then compare emergent structure to labels

### Metric status
Pearson

### Notes
This chapter is where label-blind and label-aware perspectives meet cleanly.

---

## Chapter 9. Projection onto the Template and Residual Structure

### Goal
Separate "wall-like" variance from other variance.

### Questions
- How much of each stimulus can be described as projection onto the template?
- After removing the template component, do targets still resemble each other?
- Is target cohesion entirely driven by their shared template alignment?

### Outputs
- Projection coefficients for every stimulus
- Residualized stimuli after template projection removal
- Pairwise Pearson analyses repeated on residuals
- Comparison of before/after within-group cohesion

### Label status
Label-aware

### Metric status
Pearson-centered but conceptually broader

### Notes
This chapter helps distinguish shared template structure from residual idiosyncratic variation.

---

## Chapter 10. Null Models and Permutation Tests for Pearson Structure

### Goal
Determine whether the observed Pearson-based structure is stronger than expected by chance.

### Questions
- Are targets more mutually similar than random sets of white-noise clips of the same size?
- Are distractors unusually unstructured relative to random sets?
- Is the observed separation between groups stronger than would arise from random relabeling?

### Outputs
- Null distributions from repeated random subsets
- Permutation p-values or percentile placements
- Visual comparison of empirical values against nulls

### Label status
Label-aware

### Metric status
Pearson

### Notes
This chapter provides inferential grounding for the descriptive results above.

---

# Part III. Non-Pearson Metric Analyses

## Why Part III matters

Subjects may not have been using the same dimension that defined the experimenter labels. Even if Pearson is the correct historical metric for how the stimuli were chosen, it may not be the correct metric for what information was perceptually available.

The purpose of Part III is not to replace Pearson, but to ask:

- What alternative structures exist in the stimulus set?
- Do other metrics reveal stronger grouping than Pearson?
- Do any of those groupings align with or diverge from the target/distractor labels?

---

## Chapter 11. Choosing Alternative Metrics

### Goal
Define a small, principled set of non-Pearson metrics.

### Candidate metric families

#### 1. Time-domain metrics
- Cross-correlation with lag
- Euclidean distance on normalized waveform
- Cosine similarity

#### 2. Envelope-based metrics
- Correlation of amplitude envelopes
- Distance between smoothed envelopes

#### 3. Frequency-domain metrics
- Correlation of power spectra
- Distance between spectral envelopes

#### 4. Hybrid or speech-oriented metrics
- MFCC-based distance
- Spectrotemporal similarity

### Outputs
- One short list of selected non-Pearson metrics
- Rationale for each
- Clear statement that these are exploratory structural metrics, not necessarily claims about perception

### Label status
Metric setup chapter

### Metric status
Non-Pearson

### Notes
Keep this chapter disciplined. The aim is not to try everything possible, but to choose a manageable set that each has a clear interpretation.

---

## Chapter 12. Label-Aware Group Structure Under Alternative Metrics

### Goal
Repeat core within-group and between-group analyses using non-Pearson metrics.

### Questions
- Do targets remain more similar to one another than distractors under other metrics?
- Does target/distractor separation improve or disappear under these metrics?
- Is there an alternative metric under which the groups are more distinct than under Pearson?

### Outputs
For each metric:
- within-target similarity distribution
- within-distractor similarity distribution
- between-group distribution
- effect sizes
- comparison against Pearson results

### Label status
Label-aware

### Metric status
Non-Pearson

### Notes
This chapter directly asks whether the experimenter-defined distinction is stronger under a different structural lens.

---

## Chapter 13. Label-Blind Structure Under Alternative Metrics

### Goal
Ask what intrinsic structure appears when the target/distractor labels are ignored.

### Questions
- Do natural clusters emerge under these other metrics?
- Are those clusters stronger than the Pearson-based clusters?
- Do those emergent clusters correspond to your labels or not?

### Outputs
For each metric:
- embedding plots
- clustering results
- cluster validity scores
- comparison of cluster assignments to experimenter labels

### Label status
First label-blind, then compared back to labels

### Metric status
Non-Pearson

### Notes
This is one of the most important chapters for the possibility that subjects were categorizing based on some other structure.

---

## Chapter 14. Metric Comparison: Which Dimensions Carry the Most Structure?

### Goal
Compare all metrics directly.

### Questions
- Which metric yields the strongest within-group cohesion for targets?
- Which metric yields the clearest separation between experimenter-defined groups?
- Which metric yields the strongest intrinsic clustering independent of labels?

### Outputs
- Summary table across metrics
- Ranked metrics by:
  - target cohesion
  - label separation
  - cluster strength
- Cross-metric correlations between similarity matrices

### Label status
Mixed

### Metric status
Comparative

### Notes
This chapter synthesizes Part II and Part III in a way that stays stimulus-only.

---

# Part IV. Synthesis

## Chapter 15. What Was Actually in the Stimuli?

### Goal
Provide a final synthesis of Book 1.

### Questions
- Did the stimulus set contain a strong target/distractor distinction?
- Was that distinction specific to Pearson or robust across metrics?
- Did the stimulus set contain intrinsic structure not captured by your labels?
- Was the available structure weak, noisy, multidimensional, or misleading?

### Outputs
A concise narrative summary organized around:
- structure imposed by design
- structure actually present
- structure visible only under some metrics
- implications for later subject analyses

### Label status
Mixed

### Metric status
Integrated

---

# A Cleaner Organizational View

## Axis 1. Label-aware versus label-blind

### Label-aware chapters
These ask what your stimulus labels actually produced.
- Chapter 3
- Chapter 4
- Chapter 5
- Chapter 6
- Chapter 7
- Chapter 9
- Chapter 10
- Chapter 12

### Label-blind chapters
These ask what structure exists without assuming your labels.
- Chapter 8
- Chapter 13

### Mixed synthesis chapters
- Chapter 14
- Chapter 15

---

## Axis 2. Pearson versus non-Pearson

### Pearson chapters
- Chapter 4
- Chapter 5
- Chapter 6
- Chapter 7
- Chapter 8
- Chapter 9
- Chapter 10

### Non-Pearson chapters
- Chapter 11
- Chapter 12
- Chapter 13

### Comparative chapters
- Chapter 14
- Chapter 15

---

# A Practical Minimal Version of Book 1

If you want a reduced version first before expanding into the full book, this would be the most efficient core:

## Core Book 1

### Chapter A. Stimulus inventory and preprocessing
Document what the stimuli are and how they were produced.

### Chapter B. Pearson to template distribution
Show where targets and distractors sit relative to the full candidate pool.

### Chapter C. Pairwise Pearson structure
Compare target-target, distractor-distractor, and target-distractor similarity.

### Chapter D. Label-blind Pearson geometry
Embedding and clustering under Pearson.

### Chapter E. Residual structure after removing template projection
Ask whether target similarity is just shared alignment to "wall".

### Chapter F. One or two alternative metrics
Repeat the pairwise and label-blind analyses with a small number of non-Pearson metrics.

### Chapter G. Synthesis
State what information was actually present in the stimuli.

This shorter version would already give you a very strong Book 1.

---

# A Possible Internal Logic for Writing Book 1

A clean narrative order would be:

1. **Define the stimulus set**
2. **Describe what the Pearson-based selection did**
3. **Characterize within-group and between-group structure under Pearson**
4. **Ask whether labels reflect intrinsic structure under Pearson**
5. **Remove the template component and test what remains**
6. **Repeat key analyses under alternative metrics**
7. **Conclude what dimensions of information were actually available in the stimuli**

This order should keep the reader oriented.

---

# Recommendation for How to Proceed Next

The best next step is probably to turn this plan into a more operational outline with, for each chapter:

- exact input data
- exact analyses
- exact plots
- exact numerical summaries
- exact interpretation questions

That would make Book 1 fully executable rather than just conceptual.
