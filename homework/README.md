# Homework: model-based differential expression on JNK-deficient liver

In class we analysed a multi-factor experiment (genotypes × treatments) three ways and saw
why fitting **one model with interaction terms** is cleaner than running many pairwise
tests and clustering the results. Now you will do the same analysis on published data.

## The data

Vernia *et al.* (2014) *Cell Metabolism* 20:512–525,
"The PPARα-FGF21 hormone axis contributes to metabolic regulation by the hepatic JNK
signaling pathway" ([doi:10.1016/j.cmet.2014.06.010](https://doi.org/10.1016/j.cmet.2014.06.010),
GEO [GSE55190](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55190)).

Mouse liver RNA-seq, 24 samples:

| Factor | Levels |
|---|---|
| genotype | `WT` (Alb-Cre), `Jnk1_LKO` (liver Jnk1 knockout), `Jnk2_LKO` (liver Jnk2 knockout), `Jnk1_Jnk2_LKO` (both) |
| diet | `chow`, `HFD` (high-fat diet, 16 weeks) |
| replicates | 3 mice per genotype × diet |

Files in [`data/`](data/):

* `GSE55190_counts.tsv.gz`: gene-level read counts (rows = genes; columns `gene_name`,
  `gene_type`, `length`, then one column per sample). Reads were quantified with Salmon
  against GENCODE mouse M36 and summed to genes.
* `GSE55190_samples.tsv`: one row per sample: `genotype`, `diet`, `replicate`, GEO and SRA
  accessions.

The paper's main claim: hepatic JNK represses PPARα, so JNK-deficient liver shows higher
expression of PPARα targets (fatty-acid oxidation, ketogenesis, *Fgf21*), especially on
the high-fat diet. Keep this in mind; you will test it.

## What to hand in

A [marimo](https://marimo.io) notebook (start from [`hw_starter.py`](hw_starter.py)) with:

* a **markdown cell before every code cell** explaining what the code does,
* your figures and tables,
* short written answers to the questions below (a few sentences each).

You may use PyDESeq2 (Python, used in class) or DESeq2 (R). Use FDR < 5% unless told
otherwise, and say which thresholds you used.

**Gene filter:** keep genes whose highest genotype × diet group-mean TPM (`GSE55190_tpm.tsv.gz`)
is at least **3**. In class we used 6 for strongly induced stimulation responses; metabolic
regulation involves many moderately expressed genes, so the threshold is lower here. (With 6,
*Mapk8*, which encodes JNK1 itself, would be filtered out.)

## Part 1: look at the data (notebooks 1 and 4)

1. Plot library sizes. Normalize for depth (size factor = library size / median library
   size, round to keep integers).
2. Make **replicate-vs-replicate scatter plots** for two WT chow mice on the count scale (log
   axes). Do you see the funnel? Where does Poisson noise dominate?
3. Plot **variance vs mean** across genes within one group (e.g. WT chow). Fit a curve. How
   much of the gene-to-gene variation in variance does the mean explain?

*Question:* why would a t-test on each gene, with 3 vs 3 mice, be a poor way to analyse
this experiment?

## Part 2: one gene, as a linear model (notebook 2)

4. For *Fgf21*, compare WT chow vs WT HFD with a t-test on log2 normalized counts, and with
   a linear model `y ~ diet`. Show that the t statistic and p-value are the same.

## Part 3: WT only

5. Fit DESeq2 to the 6 WT samples with `~ diet` (chow as reference). How many genes go up
   and down on HFD? Show a volcano plot and a few top genes.

## Part 4: one model for the whole experiment (as in class)

6. Fit DESeq2 to all 24 samples with

   ```
   ~ genotype + diet + genotype:diet        (references: WT, chow)
   ```

   Write a table explaining what each coefficient means. For example, `genotype[Jnk1_LKO]`
   is the Jnk1 knockout vs WT **on chow**, and `genotype[Jnk1_LKO]:diet[HFD]` is how much
   the HFD response changes in the Jnk1 knockout.
7. Plot the **dispersion estimates** (gene-wise, trend, final) against mean expression.
8. **Classify genes by which knockouts change them**, using signed calls (+1 / 0 / −1) on
   the coefficients:
   * *baseline pattern*: a 3-digit code for `genotype[Jnk1_LKO]`, `genotype[Jnk2_LKO]`,
     `genotype[Jnk1_Jnk2_LKO]` (e.g. `001` = changed only when both JNKs are missing)
   * *diet-response pattern*: the same 3-digit code for the `genotype:diet[HFD]` interactions
   * draw a heatmap of the genes, split by pattern and clustered within each pattern
9. Cross the WT HFD response (Part 3: up / not changed / down) with the knockout patterns.
   Which kinds of diet-responsive genes depend on JNK?

*Questions:*
* Are there more genes changed in the double knockout than in either single knockout? What
  does a gene with pattern `001` tell you about Jnk1 and Jnk2?
* Pick five PPARα targets (e.g. *Cyp4a14*, *Cyp4a10*, *Acot1*, *Hmgcs2*, *Fgf21*). What do
  their coefficients say about the paper's claim? On chow, on HFD, or both?

## Part 5: the classical way (as in class)

10. Test **every group against WT chow** separately (7 tests, `~ condition`, only the 6
    samples involved, with the same gene filter and FDR). Take the union of DE genes and
    cluster their fold changes with **k-means**.
11. Compare with Part 4:
    * how many genes does each approach find for the same question (e.g. HFD effect in WT)?
    * draw the k-means heatmap next to your pattern heatmap, with the model pattern as a
      color bar. Do clusters match patterns?
    * re-run k-means with a few random seeds. How many genes change cluster?
    * "induced in WT but not in the knockout" (comparing two significance calls) vs a
      significant interaction term: how many genes does each call, and how many overlap?
      Plot two genes where they disagree.

## Part 6: does the model fit? (as in class)

12. For a few genes, plot **observed vs model-predicted** normalized counts for all 24
    samples, for the full model and for a model **without the interaction**
    (`~ genotype + diet`). Add a residual plot (log2 observed / predicted) by group.
13. For how many genes does dropping the interaction make the fit clearly worse?

*Question:* the full model has one parameter per group. What does a good predicted-vs-observed
plot tell you, and what does it not?

## Bonus

* Use a **likelihood ratio test** (full vs `~ genotype + diet`) to ask, per gene, whether
  JNK changes the diet response at all, testing all three interactions together.
* Normalize with DESeq2's median-of-ratios size factors instead of library size. Does it
  change your conclusions?

## Class notebooks

The notebooks from class are in the parent folder:
[`01_distributions.py`](../01_distributions.py),
[`02_ttest_is_lm.py`](../02_ttest_is_lm.py),
[`03_counts_and_glm.py`](../03_counts_and_glm.py),
[`04_variance_estimation.py`](../04_variance_estimation.py).
Run them with `marimo edit <file>` (see the [main README](../README.md)).
