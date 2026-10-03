# Bootcamp session 4: bulk RNA-seq, why DE uses models

Marimo notebooks, in order:

1. `01_distributions.py`: arrays vs RNA-seq (GAPDH/ACTB histograms, depth normalization, mean vs variance)
2. `02_ttest_is_lm.py`: the t-test as a linear model; adding a batch term
3. `03_counts_and_glm.py`: simulated qPCR/array/RNA-seq, negative binomial, GLM preview
4. `04_variance_estimation.py`: why per-gene variances need help; variance-vs-mean curve; shrinkage (limma eBayes / DESeq2 idea)

```bash
# with uv (installs dependencies from the notebook header)
uvx marimo edit --sandbox 01_distributions.py

# or with an existing environment
pip install marimo numpy scipy statsmodels pandas matplotlib
marimo edit .   # opens a file browser for all notebooks
```

## Data
* Microarray: GEO GSE8658 (moDCs, HG-U133 Plus 2), downloaded on first run into `data/array/`.
  `data/array/GPL570_probe2symbol.tsv.gz` is the probe → symbol map from GPL570.
* RNA-seq: lab DC raw-count tables, not in this repo. Point `BULK_RNASEQ_DIR` at the folder
  (default `~/Dropbox_umms/projects/carol_chemokine_data`).
* Gene lengths for TPM: `data/genes/gencode_v44_gene_length.tsv.gz` (merged exon length per gene, GENCODE v44 basic).
