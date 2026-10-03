# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "scipy", "statsmodels", "pandas", "matplotlib"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. How are expression measurements distributed?

    Two comparisons between microarrays and RNA-seq, both from human monocyte-derived
    dendritic cells:

    1. **One gene across many samples.** Are GAPDH and ACTB roughly normal?
    2. **Mean vs variance.** How much does a gene's spread depend on its expression level?

    The answers tell us which models fit: linear models (normal errors, one variance) for
    arrays, and count models (Poisson / negative binomial GLMs) for RNA-seq.

    The first cell loads the libraries, sets the data locations, and defines a small plotting
    helper: a histogram with a normal curve of the same mean and SD.
    """)
    return


@app.cell
def _():
    import gzip
    import io
    import os
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    ARRAY_COLOR, SEQ_COLOR = "#55A868", "#C44E52"
    EXP_COLORS = {"UVB": "#C44E52", "TNF_IL32": "#8172B3", "public": "#937860"}
    GENES = ["GAPDH", "ACTB"]

    DATA_DIR = Path(__file__).parent / "data"
    RNASEQ_DIR = Path(os.environ.get("BULK_RNASEQ_DIR", "~/Dropbox_umms/projects/carol_chemokine_data")).expanduser()


    def hist_with_normal(ax, values, color, title, xlabel, bins=20, by=None, colors=None):
        """Histogram (optionally stacked by group) with a fitted normal curve."""
        values = np.asarray(values, dtype=float)
        edges = np.histogram_bin_edges(values, bins=bins)
        if by is None:
            ax.hist(values, bins=edges, color=color, density=True)
        else:
            groups = list(colors)
            ax.hist([values[np.asarray(by) == g] for g in groups], bins=edges, stacked=True,
                     density=True, color=[colors[g] for g in groups], label=groups)
        x = np.linspace(edges[0], edges[-1], 300)
        ax.plot(x, stats.norm.pdf(x, values.mean(), values.std()), "k-", lw=1.5)
        p = stats.shapiro(values).pvalue
        ax.set(title=f"{title}\nShapiro p = {p:.2g}", xlabel=xlabel, yticks=[])


    def two_gene_figure():
        return plt.subplots(1, 2, figsize=(11, 3.4))

    return (
        ARRAY_COLOR,
        DATA_DIR,
        EXP_COLORS,
        GENES,
        RNASEQ_DIR,
        SEQ_COLOR,
        gzip,
        hist_with_normal,
        io,
        mo,
        np,
        pd,
        plt,
        sm,
        stats,
        two_gene_figure,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Microarray data

    [GSE8658](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8658): monocytes and
    monocyte-derived DCs from 6 donors at several time points and treatments, 63
    Affymetrix HG-U133 Plus 2 arrays. The deposited values are already **normalized**
    (GC-RMA) on the linear scale. Arrays are analyzed as log2 intensity, so we take log2.
    Each gene is represented by its highest-expressed probe.

    The code downloads the matrix from GEO on first run and caches it in `data/array/`.
    """)
    return


@app.cell
def _(DATA_DIR, gzip, io, np, pd, urllib):
    _path = DATA_DIR / "array" / "GSE8658_series_matrix.txt.gz"
    if not _path.exists():
        _path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8658/matrix/GSE8658_series_matrix.txt.gz", _path
        )
    _lines = gzip.open(_path, "rt").read().splitlines()
    _titles = [t.strip('"') for t in next(l for l in _lines if l.startswith("!Sample_title")).split("\t")[1:]]
    _start = _lines.index("!series_matrix_table_begin")
    _probes = pd.read_csv(io.StringIO("\n".join(_lines[_start + 1 : -1])), sep="\t", index_col=0)
    _probes.columns = _titles
    _probes = np.log2(_probes)

    _symbol = pd.read_csv(DATA_DIR / "array" / "GPL570_probe2symbol.tsv.gz", sep="\t", index_col=0)["Gene Symbol"]
    _symbol = _symbol.reindex(_probes.index)
    _ok = _symbol.notna() & ~_symbol.str.contains("///", regex=False, na=True)
    _best = _probes[_ok].mean(axis=1).groupby(_symbol[_ok]).idxmax()
    arr = _probes.loc[_best.values].set_axis(_best.index)


    def _group(title):
        if title.startswith("MC"):
            return "monocyte"
        time, donor, *treat = title.split()
        return f"{time} DC " + (" ".join(treat) if treat else "untreated")


    arr_groups = pd.Series([_group(t) for t in _titles], index=_titles)
    arr.shape
    return arr, arr_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RNA-seq data

    Raw read counts from three monocyte-derived DC experiments: two from the lab
    (UVB/IFNβ and TNF/IL32) and one public stimulation set. Libraries with fewer than
    1 M reads are dropped.

    The plot shows the **library size** (total reads) of each sample. The public samples
    were sequenced about 10× deeper than the lab samples. Keep that in mind for the next
    histograms.
    """)
    return


@app.cell
def _(EXP_COLORS, RNASEQ_DIR, mo, np, pd, plt):
    mo.stop(
        not RNASEQ_DIR.exists(),
        mo.callout(mo.md(f"RNA-seq folder not found: `{RNASEQ_DIR}`. Set `BULK_RNASEQ_DIR`."), kind="warn"),
    )
    _tables = {
        "UVB": pd.read_csv(RNASEQ_DIR / "DCs_UVB/DC_uvb_meta_raw.tsv", sep="\t", index_col=0),
        "TNF_IL32": pd.read_csv(RNASEQ_DIR / "DCs_TNF_IL32/DC_TNF_IL32_raw.tsv", sep="\t").set_index("gene"),
        "public": pd.read_csv(RNASEQ_DIR / "DC_public_data/DC_pd_raw.tsv", sep="\t", index_col=0)
        .round()
        .astype(int)
        .add_prefix("pub_"),
    }
    _counts = pd.concat(_tables.values(), axis=1, join="inner")
    _exp = np.concatenate([[k] * t.shape[1] for k, t in _tables.items()])
    _samples = pd.DataFrame({"experiment": _exp, "lib_size": _counts.sum().values}, index=_counts.columns)
    _samples["group"] = [
        f"{e}: " + (c[4:].rsplit("_", 1)[0] if e == "public" else c.split("_", 1)[1])
        for c, e in zip(_counts.columns, _exp)
    ]
    _keep = _samples.lib_size >= 1e6
    counts, samples = _counts.loc[:, _keep], _samples[_keep]

    _fig, _ax = plt.subplots(figsize=(12, 2.8))
    _ax.bar(range(len(samples)), samples.lib_size / 1e6, color=samples.experiment.map(EXP_COLORS))
    _ax.set(xticks=[], xlabel=f"{len(samples)} samples", ylabel="library size (M reads)")
    for _e, _c in EXP_COLORS.items():
        _ax.bar(0, 0, color=_c, label=_e)
    _ax.legend(frameon=False, title="experiment")
    _fig.tight_layout()
    _fig
    return counts, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GAPDH and ACTB on the microarray

    Histogram of each gene's normalized log2 intensity across all 63 arrays. The black
    curve is a normal distribution with the same mean and SD, which is what a linear
    model assumes. The Shapiro–Wilk p-value tests normality (small p means not normal).
    """)
    return


@app.cell
def _(ARRAY_COLOR, GENES, arr, hist_with_normal, two_gene_figure):
    _fig, _ax = two_gene_figure()
    for _a, _g in zip(_ax, GENES):
        hist_with_normal(_a, arr.loc[_g], ARRAY_COLOR, f"Array {_g} (n={arr.shape[1]})", "normalized log2 intensity")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same genes in RNA-seq: raw counts

    Same plot for raw read counts across all RNA-seq samples, stacked by experiment.
    """)
    return


@app.cell
def _(
    EXP_COLORS,
    GENES,
    SEQ_COLOR,
    counts,
    hist_with_normal,
    samples,
    two_gene_figure,
):
    _fig, _ax = two_gene_figure()
    for _a, _g in zip(_ax, GENES):
        hist_with_normal(_a, counts.loc[_g], SEQ_COLOR, f"RNA-seq {_g}, raw counts (n={counts.shape[1]})",
                         "reads", bins=25, by=samples.experiment, colors=EXP_COLORS)
    _ax[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Raw counts mostly track **sequencing depth**: the deep public libraries form their own
    cluster. Counts from libraries of different size are not comparable.

    ## Normalizing for sequencing depth

    Use the simplest correction. Each sample's **size factor** is its library size
    relative to the median library size, and each count is divided by it:

    $$s_j = \frac{N_j}{\operatorname{median}_k N_k}, \qquad \tilde y_{gj} = \operatorname{round}\!\left(\frac{y_{gj}}{s_j}\right)$$

    Every library is rescaled to the median depth. Rounding keeps the normalized values
    integers, so they are still count data that a Poisson or negative binomial model can
    describe. (DESeq2's median-of-ratios size factors are a more robust version of the
    same idea.)
    """)
    return


@app.cell
def _(counts, mo, np, samples):
    size_factor = samples.lib_size / samples.lib_size.median()
    norm_counts = np.rint(counts / size_factor).astype(int)
    mo.md(
        f"Size factors range from **{size_factor.min():.2f}** to **{size_factor.max():.2f}** "
        f"(median library size {samples.lib_size.median() / 1e6:.1f} M reads)."
    )
    return (norm_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GAPDH and ACTB in RNA-seq after depth normalization

    Same histograms, now with depth-normalized integer counts.
    """)
    return


@app.cell
def _(
    EXP_COLORS,
    GENES,
    SEQ_COLOR,
    hist_with_normal,
    norm_counts,
    samples,
    two_gene_figure,
):
    _fig, _ax = two_gene_figure()
    for _a, _g in zip(_ax, GENES):
        hist_with_normal(_a, norm_counts.loc[_g], SEQ_COLOR, f"RNA-seq {_g}, normalized counts",
                         "normalized reads", bins=25, by=samples.experiment, colors=EXP_COLORS)
    _ax[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to notice**

    * **ACTB:** the depth clusters are gone, but the counts are still right-skewed with a
      long upper tail. Biological variation is multiplicative, so counts are roughly
      log-normal, not normal. The array, which reports log intensity, looks normal.
      Part of ACTB's tail is the public experiment (brown), which sits higher than the
      lab samples: a between-experiment difference that a model would handle with a
      batch term.
    * **GAPDH:** varies very little between samples (CV ≈ 0.3). With that little spread a
      log-normal is hard to tell from a normal, so GAPDH happens to look fine.
    * Do **not** take log2 of the counts here: that would make them look normal again and
      hide the difference we are trying to show.

    ## Is that true for all genes? Skewness across samples

    For every expressed gene, compute the skewness of its values across all samples:
    0 for a symmetric distribution, > 0 for a long right tail. Array genes: above the 30th
    percentile of mean intensity (drops background probes). RNA-seq genes: median
    normalized count > 10.
    """)
    return


@app.cell
def _(ARRAY_COLOR, SEQ_COLOR, arr, norm_counts, np, plt, stats):
    _am = arr.mean(axis=1)
    _sk_a = stats.skew(arr[_am > _am.quantile(0.3)].values, axis=1)
    _sk_s = stats.skew(norm_counts[norm_counts.median(axis=1) > 10].values, axis=1)

    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _bins = np.linspace(-3, 6, 80)
    _ax.hist(np.clip(_sk_a, -3, 6), bins=_bins, alpha=0.6, density=True, color=ARRAY_COLOR,
             label=f"array, log2 intensity ({len(_sk_a):,} genes, median {np.median(_sk_a):.2f})")
    _ax.hist(np.clip(_sk_s, -3, 6), bins=_bins, alpha=0.6, density=True, color=SEQ_COLOR,
             label=f"RNA-seq, normalized counts ({len(_sk_s):,} genes, median {np.median(_sk_s):.2f})")
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set(xlabel="skewness across samples", yticks=[])
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Caveat: both datasets pool several conditions, so part of the skew is biology
    (induced genes). That affects both technologies equally.

    ## Mean vs variance within replicates

    To see pure noise, use one homogeneous group per technology: same cell state,
    different donors. Pick the groups here (only groups with at least 4 samples).
    """)
    return


@app.cell
def _(arr_groups, mo, samples):
    _a = arr_groups.value_counts()
    _s = samples.group.value_counts()
    arr_group = mo.ui.dropdown({f"{g} (n={n})": g for g, n in _a[_a >= 4].items()},
                               value=f"5d DC untreated (n={_a['5d DC untreated']})", label="array group")
    seq_group = mo.ui.dropdown({f"{g} (n={n})": g for g, n in _s[_s >= 4].items()},
                               value=f"public: Ctrl (n={_s['public: Ctrl']})", label="RNA-seq group")
    mo.hstack([arr_group, seq_group], justify="start", gap=2)
    return arr_group, seq_group


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For every gene, compute its mean and SD across the replicates of the chosen group:
    arrays in log2 intensity, RNA-seq in normalized counts. The black line is a lowess
    trend. The right panel divides each trend by its median SD and plots it against
    expression percentile, so both technologies share the same axes.
    """)
    return


@app.cell
def _(
    ARRAY_COLOR,
    SEQ_COLOR,
    arr,
    arr_group,
    arr_groups,
    norm_counts,
    np,
    plt,
    samples,
    seq_group,
    sm,
):
    _A = arr.loc[:, arr_groups == arr_group.value]
    _S = norm_counts.loc[:, samples.group == seq_group.value]
    _S = _S[(_S > 0).all(axis=1)]

    _a_m, _a_sd = _A.mean(axis=1), _A.std(axis=1)
    _s_m, _s_sd = _S.mean(axis=1), _S.std(axis=1)
    _s_m, _s_sd = _s_m[_s_sd > 0], _s_sd[_s_sd > 0]
    _a_fit = sm.nonparametric.lowess(_a_sd, _a_m, frac=0.2)
    _s_fit = 10 ** sm.nonparametric.lowess(np.log10(_s_sd), np.log10(_s_m), frac=0.2)

    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4))
    _ax[0].scatter(_a_m, _a_sd, s=2, alpha=0.15, color=ARRAY_COLOR)
    _ax[0].plot(*_a_fit.T, "k-", lw=2)
    _ax[0].set(xlabel="mean log2 intensity", ylabel="SD (log2 intensity)", ylim=(0, np.quantile(_a_sd, 0.999)),
               title=f"Array: {arr_group.value} (n={_A.shape[1]})")

    _ax[1].scatter(_s_m, _s_sd, s=2, alpha=0.15, color=SEQ_COLOR)
    _ax[1].plot(*_s_fit.T, "k-", lw=2, label="trend")
    _g = np.logspace(np.log10(_s_m.min()), np.log10(_s_m.max()), 100)
    _ax[1].plot(_g, np.sqrt(_g), "k--", lw=1, label="Poisson: SD = √mean")
    _ax[1].set(xscale="log", yscale="log", xlabel="mean normalized count", ylabel="SD (counts)",
               title=f"RNA-seq: {seq_group.value} (n={_S.shape[1]})")
    _ax[1].legend(frameon=False, fontsize=8)

    for _m, _sd, _fit, _c, _lab in ((_a_m, _a_sd, _a_fit, ARRAY_COLOR, "array (log2 intensity)"),
                                    (_s_m, _s_sd, _s_fit, SEQ_COLOR, "RNA-seq (counts)")):
        _pct = np.searchsorted(np.sort(_m.values), _fit[:, 0]) / len(_m) * 100
        _ax[2].plot(_pct, _fit[:, 1] / np.median(_sd), color=_c, lw=2.5, label=_lab)
    _ax[2].axhline(1, color="gray", lw=0.5)
    _ax[2].set(yscale="log", xlabel="expression percentile", ylabel="SD trend / median SD",
               title="How much does SD depend on level?")
    _ax[2].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Take-home**

    * **Arrays:** the SD of log2 intensity changes only a few-fold across the whole
      expression range. One variance per gene (or a gentle trend) is a good model, so a
      **linear model** (t-test, `limma`) fits.
    * **RNA-seq:** the SD of counts grows with the mean over orders of magnitude. Low-count
      genes sit near the Poisson line (sampling noise); higher-count genes rise above it
      (biological variation, $\operatorname{Var} = \mu + \phi\mu^2$). Variance is a
      *function of the mean*, and values are skewed integers, so we use a **count GLM**
      (negative binomial: DESeq2, edgeR).
    """)
    return


if __name__ == "__main__":
    app.run()
