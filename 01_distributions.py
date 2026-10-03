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

    * **GAPDH and ACTB are poor test cases.** They have thousands of counts per sample,
      and a negative binomial (NB) with a large mean is nearly symmetric: its skewness
      approaches $2\sqrt{\phi}$, about 0.2–0.6 for typical dispersions. So a perfectly
      NB gene can look normal.
    * ACTB's long right tail is mostly the public experiment (brown) sitting higher than
      the lab samples. That is a difference between experiments, not the shape of the
      count distribution.
    * Do **not** take log2 of the counts here: that would make them look normal and hide
      the point.

    ## Stable genes at lower expression

    To see the count distribution itself, we need genes that do **not** respond to the
    stimulations. These six were selected from the data:

    * expression differs little between conditions (condition explains < 40% of the
      variance; with 22 conditions and 81 samples, pure noise alone gives about 26%)
    * no difference between the three experiments (< 5% of the variance)
    * spread close to sampling noise (variance / mean < 1.6, where Poisson = 1)

    The table recomputes these numbers. Each histogram shows the gene's depth-normalized
    counts across all 81 samples, with two fits using the same mean and variance: the
    **negative binomial** (black) and the **normal** (blue).
    """)
    return


@app.cell
def _(mo, norm_counts, np, pd, plt, samples, stats):
    STABLE_GENES = ["ZNF483", "NAT14", "SLC25A53", "N6AMT1", "METTL2B", "CUL5"]


    def _var_explained(x, labels):
        group_means = x.groupby(labels.values).transform("mean")
        return 1 - ((x - group_means) ** 2).sum() / ((x - x.mean()) ** 2).sum()


    _rows = []
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for _ax, _g in zip(_axes.flat, STABLE_GENES):
        _y = norm_counts.loc[_g]
        _m, _v = _y.mean(), _y.var()
        _phi = max((_v - _m) / _m**2, 1e-6)
        _w = max(1, int(np.ceil((_y.max() + 1) / 30)))
        _ax.hist(_y, bins=np.arange(-0.5, _y.max() + _w + 0.5, _w), density=True, color="#C44E52", alpha=0.7)
        _k = np.arange(0, _y.max() + 1)
        _ax.plot(_k, stats.nbinom.pmf(_k, 1 / _phi, 1 / (1 + _phi * _m)), "o" if len(_k) < 40 else "-",
                 ms=3, lw=1.5, color="k", label="negative binomial")
        _x = np.linspace(min(_m - 3.5 * np.sqrt(_v), -0.5), _y.max() + _w, 300)
        _ax.plot(_x, stats.norm.pdf(_x, _m, np.sqrt(_v)), "-", color="#4C72B0", lw=2, label="normal")
        _ax.axvline(0, color="gray", lw=0.5)
        _ax.set(title=f"{_g}  (mean {_m:.1f})", xlabel="normalized count", yticks=[])
        _rows.append({
            "gene": _g,
            "mean": round(_m, 1),
            "variance / mean": round(_v / _m, 2),
            "zeros": int((_y == 0).sum()),
            "% var. condition": round(100 * _var_explained(_y, samples.group)),
            "% var. experiment": round(100 * _var_explained(_y, samples.experiment)),
            "skewness": round(stats.skew(_y), 2),
            "Shapiro p": f"{stats.shapiro(_y).pvalue:.1g}",
        })
    _axes.flat[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to notice**

    * At a mean of 1–4 counts the data are a handful of integers piled against zero. The
      NB fits; the normal is skewed the wrong way and puts probability on **negative
      counts**.
    * By about 10 counts the NB and the normal nearly coincide, and from there up the
      shape alone can't tell them apart.
    * Below roughly 10 counts the normal fails. The next section asks how many genes
      fall in that range, in TPM terms.

    ## At what expression level does a normal stop fitting the counts?

    Counts are hard to interpret as an expression level: the same gene gives 10× more
    reads in a library sequenced 10× deeper, and long genes collect more reads than short
    ones. **TPM** (transcripts per million) corrects for both:

    $$\text{TPM}_g = 10^6 \cdot \frac{y_g / L_g}{\sum_h y_h / L_h}$$

    where $L_g$ is the gene's length (merged exon length from GENCODE v44, stored in
    `data/genes/`).

    For each gene in a replicate group:

    1. Compute its mean TPM and its mean count $\mu$.
    2. Assume its counts follow a negative binomial $\text{NB}(\mu, \phi)$, with $\phi$ the
       group's dispersion (median moment estimate among genes with mean count > 100).
    3. Compare that NB with a normal of the same mean and variance. The **misfit** is
       the largest gap between their cumulative distributions (0 = identical; 0.05 = the
       normal gets some cumulative probability wrong by 5 percentage points).

    We do this for a deeply sequenced group (public controls, ~27 M reads) and a
    shallow one (lab UVB mock, ~3 M reads). Use the slider to set the misfit you would
    call "poorly fit".
    """)
    return


@app.cell
def _(mo):
    misfit_cut = mo.ui.slider(0.02, 0.2, step=0.01, value=0.05, label="misfit threshold", show_value=True)
    misfit_cut
    return (misfit_cut,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The left panel shows, for genes binned by TPM, the fraction whose counts are poorly
    fit by a normal. The dashed lines mark the TPM below which most genes (> 50%) are
    misfit. The right panel plots the same fractions against mean count; the two groups
    collapse onto one curve, because the fit depends only on the count.
    """)
    return


@app.cell
def _(DATA_DIR, counts, misfit_cut, mo, np, pd, plt, stats):
    _length = pd.read_csv(DATA_DIR / "genes" / "gencode_v44_gene_length.tsv.gz", sep="\t", index_col=0)["exon_length"]
    _groups = {
        "deep: public controls": [c for c in counts.columns if c.startswith("pub_Ctrl")],
        "shallow: lab UVB mock": [c for c in counts.columns if c.endswith("_M")],
    }
    _colors = dict(zip(_groups, ["#937860", "#C44E52"]))


    def _misfit_curve(phi):
        """Max CDF gap between NB(mu, phi) and a moment-matched normal, on a grid of mu."""
        grid = np.logspace(-2, 4, 300)
        gaps = []
        for mu in grid:
            sd = np.sqrt(mu + phi * mu**2)
            k = np.arange(0, int(mu + 10 * sd) + 2)
            nb = stats.nbinom.cdf(k, 1 / phi, 1 / (1 + phi * mu))
            gaps.append(np.max(np.abs(nb - stats.norm.cdf((k + 0.5 - mu) / sd))))
        return grid, np.array(gaps)


    _fig, _ax = plt.subplots(1, 2, figsize=(14, 4))
    _tpm_bins = np.logspace(-2, 3, 26)
    _rows = []
    for _name, _cols in _groups.items():
        _y = counts.loc[counts.index.intersection(_length.index), _cols]
        _y = _y[_y.sum(axis=1) > 0]
        _mu = (_y / (_y.sum() / _y.sum().mean())).mean(axis=1)
        _rpk = _y.div(_length[_y.index], axis=0)
        _tpm = (_rpk / _rpk.sum() * 1e6).mean(axis=1)
        _phi = np.median(((_y.var(axis=1) - _mu) / _mu**2)[_mu > 100])
        _grid, _gap = _misfit_curve(_phi)
        _bad = np.interp(np.log10(_mu), np.log10(_grid), _gap) > misfit_cut.value

        _bin = np.digitize(_tpm, _tpm_bins)
        _frac = pd.Series(_bad).groupby(_bin).mean()
        _centers = np.sqrt(_tpm_bins[:-1] * _tpm_bins[1:])
        _ok = (_frac.index > 0) & (_frac.index < len(_tpm_bins))
        _x = _centers[_frac.index[_ok] - 1]
        _ax[0].plot(_x, 100 * _frac[_ok].values, "o-", color=_colors[_name], label=_name)
        _above = _x[100 * _frac[_ok].values <= 50]
        _tpm_cut = _above.min() if len(_above) else np.nan
        _ax[0].axvline(_tpm_cut, color=_colors[_name], ls="--", lw=1)

        _cbins = np.logspace(-2, 4, 31)
        _cfrac = pd.Series(_bad).groupby(np.digitize(_mu, _cbins)).mean()
        _cok = (_cfrac.index > 0) & (_cfrac.index < len(_cbins))
        _ax[1].plot(np.sqrt(_cbins[:-1] * _cbins[1:])[_cfrac.index[_cok] - 1], 100 * _cfrac[_cok].values,
                    "o-", color=_colors[_name], label=_name)
        _rows.append({
            "group": _name,
            "mean depth (M reads)": round(_y.sum().mean() / 1e6, 1),
            "dispersion φ": round(_phi, 3),
            "TPM below which most genes misfit": round(_tpm_cut, 2),
            "% of detected genes below it": round(100 * (_tpm < _tpm_cut).mean()),
            "% misfit overall": round(100 * _bad.mean()),
        })
    _ax[0].set(xscale="log", xlabel="mean TPM", ylabel="% of genes poorly fit by a normal",
               title="Normal misfit vs expression level (TPM)")
    _ax[1].set(xscale="log", xlabel="mean count", ylabel="% of genes poorly fit by a normal",
               title="Same, vs mean count")
    for _a in _ax:
        _a.legend(frameon=False, fontsize=8)
        _a.set_ylim(-3, 103)
    _fig.tight_layout()
    mo.vstack([_fig, mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to notice**

    * In the deep libraries the normal breaks down below roughly 0.3 TPM; in the shallow
      ones below roughly 1 TPM. Either way that is about a quarter of the detected genes.
    * The TPM cutoff is not a property of the gene: sequencing 10× deeper lowers it,
      because what matters is the **count**. The right panel shows this.
    * Above a few TPM a normal describes each gene's counts well. That alone does not
      make a linear model fine: the variance still depends on the mean (next section),
      and an experiment covers both high- and low-count genes.

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
