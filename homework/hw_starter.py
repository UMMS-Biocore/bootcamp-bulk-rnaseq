# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "scipy", "pandas", "matplotlib", "statsmodels", "pydeseq2", "scikit-learn"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework: JNK-deficient liver (Vernia et al. 2014)

    Your name:

    Instructions are in [`README.md`](README.md). Keep the rule from class: **a markdown cell
    before every code cell** saying what the code does, and short answers in markdown.

    The first cell loads the libraries.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    DATA = Path(__file__).parent / "data"
    return DATA, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data

    `counts`: genes × samples (integer read counts). `genes`: gene name, type and length for
    each gene id. `samples`: one row per sample with genotype, diet and replicate. Genotype
    and diet are categorical with **WT** and **chow** as reference levels.
    """)
    return


@app.cell
def _(DATA, pd):
    _raw = pd.read_csv(DATA / "GSE55190_counts.tsv.gz", sep="\t", index_col=0)
    genes = _raw[["gene_name", "gene_type", "length"]]
    counts = _raw.drop(columns=["gene_name", "gene_type", "length"])
    samples = pd.read_csv(DATA / "GSE55190_samples.tsv", sep="\t", index_col=0).loc[counts.columns]
    samples["genotype"] = pd.Categorical(samples.genotype, ["WT", "Jnk1_LKO", "Jnk2_LKO", "Jnk1_Jnk2_LKO"])
    samples["diet"] = pd.Categorical(samples.diet, ["chow", "HFD"])
    counts.shape, samples.shape
    return counts, genes, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The design: samples per genotype × diet.
    """)
    return


@app.cell
def _(pd, samples):
    pd.crosstab(samples.genotype, samples.diet)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: look at the data

    *Explain what the next cell does, then write it.*
    """)
    return


@app.cell
def _():
    # TODO: library sizes, depth normalization, replicate-vs-replicate scatters, variance vs mean
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 onwards

    Add a markdown cell and a code cell for each step in the README. A helper you may want,
    the PyDESeq2 pattern used in class:

    ```python
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts_kept.T, metadata=samples[["genotype", "diet"]],
                       design="~ genotype + diet + genotype:diet")
    dds.deseq2()
    print(dds.varm["LFC"].columns)          # coefficient names (natural-log scale)
    v = np.zeros(len(dds.varm["LFC"].columns)); v[k] = 1   # test coefficient k
    st = DeseqStats(dds, contrast=v); st.summary(); st.results_df
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
