# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "statsmodels",
#     "pandas",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Why we do differential expression with models

    **Bootcamp session 4 — bulk RNA-seq**

    Plan for this notebook:

    1. **Same biology, different measurements.** What do qPCR, microarray and RNA-seq
       data look like for the *same* underlying gene expression?
    2. **The t-test is a linear model.** Comparing two group means is the same as fitting
       $y = \beta_0 + \beta_1 x + \varepsilon$ and testing $\beta_1 = 0$.
    3. **Why that matters.** Once you think in models you can add covariates (batch, sex,
       subject), and swap the error distribution to one that fits counts (GLMs).

    Sections 1, 2 and 3 are simulated so we know the truth; section 1b uses real
    dendritic-cell data (public microarrays + lab RNA-seq). Move the sliders.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: setup.** Import the libraries used throughout (numpy/pandas for data, scipy and statsmodels for tests and model fits, matplotlib for plots) and define a shared color palette.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    COLORS = {"qpcr": "#4C72B0", "array": "#55A868", "seq": "#C44E52", "A": "#4C72B0", "B": "#DD8452"}
    return COLORS, Path, mo, np, pd, plt, sm, smf, stats


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Same biology, three measurements

    We simulate one gene measured in many biological replicates. Each replicate has a
    *true* abundance $\lambda$ that varies biologically around a mean $\mu$:

    $$\lambda \sim \text{Gamma}(\text{mean}=\mu,\ \text{CV}^2=\phi)$$

    $\phi$ is the **biological coefficient of variation squared** (in RNA-seq lingo, the
    *dispersion*). Then each technology measures $\lambda$ differently:

    | Technology | What you get | Measurement model |
    |---|---|---|
    | qPCR | Ct value (continuous) | $\text{Ct} = C_0 - \log_2\lambda + \varepsilon,\ \varepsilon\sim N(0, 0.25^2)$ |
    | Microarray | log2 fluorescence (continuous) | $\log_2(\text{bg} + k\lambda) + \varepsilon$, saturating at $2^{16}$ |
    | RNA-seq | read **counts** (integers $\ge 0$) | $y \sim \text{Poisson}(\lambda)$ |

    Poisson sampling on top of Gamma biological variation gives exactly the
    **negative binomial**: $\operatorname{Var}(y) = \mu + \phi\mu^2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Two sliders: the mean expression level of the gene and its biological variability (dispersion φ). Every plot below recomputes when you move them.
    """)
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(
        steps=[0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        value=10,
        label="mean expression μ (expected reads)",
        show_value=True,
    )
    phi_slider = mo.ui.slider(
        0.01, 1.0, step=0.01, value=0.1, label="biological CV² φ (dispersion)", show_value=True
    )
    mo.hstack([mu_slider, phi_slider], justify="start", gap=2)
    return mu_slider, phi_slider


@app.cell
def _(mo):
    mo.md(r"""
    **Code: simulation helpers.** `sim_abundance` draws the true per-sample abundance λ from a Gamma distribution. The three `measure_*` functions turn λ into what each technology reports: a Ct value, a log2 array intensity, or a Poisson read count.
    """)
    return


@app.cell
def _(np):
    def sim_abundance(rng, mu, phi, size):
        """True per-sample abundance: Gamma with mean mu and CV^2 phi."""
        mu = np.asarray(mu, dtype=float)
        return rng.gamma(shape=1 / phi, scale=mu * phi, size=size)

    def measure_qpcr(rng, lam, c0=37.0, sd=0.25):
        # a reaction with ~0 template never crosses threshold: cap at 40 cycles
        return np.minimum(c0 - np.log2(np.maximum(lam, 1e-9)) + rng.normal(0, sd, lam.shape), 40.0)

    def measure_array(rng, lam, bg=150.0, k=20.0, sd=0.15):
        intensity = np.minimum(bg + k * lam, 2**16)
        return np.log2(intensity) + rng.normal(0, sd, lam.shape)

    def measure_seq(rng, lam):
        return rng.poisson(lam)

    return measure_array, measure_qpcr, measure_seq, sim_abundance


@app.cell
def _(mo):
    mo.md(r"""
    **Code: one gene, three technologies.** Draw 3000 biological replicates of the same gene, measure each with all three technologies, and plot the histograms. The summary line compares the observed count variance with the Poisson and negative-binomial predictions.
    """)
    return


@app.cell
def _(
    COLORS,
    measure_array,
    measure_qpcr,
    measure_seq,
    mo,
    mu_slider,
    np,
    phi_slider,
    plt,
    sim_abundance,
):
    _rng = np.random.default_rng(42)
    _mu, _phi = mu_slider.value, phi_slider.value
    _lam = sim_abundance(_rng, _mu, _phi, 3000)
    _ct = measure_qpcr(_rng, _lam)
    _arr = measure_array(_rng, _lam)
    _cnt = measure_seq(_rng, _lam)

    _fig, _ax = plt.subplots(1, 4, figsize=(14, 3.2))
    _ax[0].hist(_ct, bins=40, color=COLORS["qpcr"])
    _ax[0].set(title="qPCR", xlabel="Ct")
    _ax[1].hist(_arr, bins=40, color=COLORS["array"])
    _ax[1].set(title="Microarray", xlabel="log2 intensity")
    _bins = np.arange(-0.5, _cnt.max() + 1.5, max(1, int(np.ceil((_cnt.max() + 1) / 60))))
    _ax[2].hist(_cnt, bins=_bins, color=COLORS["seq"])
    _ax[2].set(title="RNA-seq (raw counts)", xlabel="reads")
    _ax[3].hist(np.log2(_cnt + 1), bins=40, color=COLORS["seq"], alpha=0.7)
    _ax[3].set(title="RNA-seq log2(count + 1)", xlabel="log2(reads + 1)")
    for _a in _ax:
        _a.set_yticks([])
    _fig.tight_layout()

    _summary = mo.md(
        f"""
    **RNA-seq counts:** mean = {_cnt.mean():.2f}, variance = {_cnt.var():.2f},
    variance / mean = {_cnt.var() / max(_cnt.mean(), 1e-9):.2f}
    (Poisson would be 1; NB predicts {1 + _phi * _mu:.2f}),
    zeros = {100 * (_cnt == 0).mean():.0f}%, distinct values = {len(np.unique(_cnt))}
    """
    )
    mo.vstack([_fig, _summary])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Things to try**

    * Set μ to 0.3–3. The counts are a handful of integers piled up at zero.
      No transformation turns that into a bell curve — `log2(x+1)` just relabels the bars.
    * Set μ to 1000+. Counts look continuous and roughly symmetric; on the log scale
      they look a lot like the qPCR/array data.
    * Increase φ. Every technology gets wider (that's biology), but for counts the
      variance/mean ratio grows with μ: extra-Poisson variance is $\phi\mu^2$.
    * Low μ on the array: the signal sinks into background. High μ: saturation.
      Arrays have their own problems, they're just *continuous* problems.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Mean–variance relationship across genes

    One gene is anecdotal. Now simulate thousands of genes with means spanning five orders
    of magnitude, each measured in *n* replicates, and plot how spread depends on level.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Number of replicates per gene and a common dispersion φ for all simulated genes.
    """)
    return


@app.cell
def _(mo):
    n_rep_slider = mo.ui.slider(3, 30, value=6, label="replicates per gene", show_value=True)
    phi_genes_slider = mo.ui.slider(
        0.01, 0.5, step=0.01, value=0.1, label="dispersion φ (all genes)", show_value=True
    )
    mo.hstack([n_rep_slider, phi_genes_slider], justify="start", gap=2)
    return n_rep_slider, phi_genes_slider


@app.cell
def _(mo):
    mo.md(r"""
    **Code: mean–variance across genes.** Simulate 3000 genes with means from 0.1 to 10,000 reads, measure each in *n* replicates, then plot per-gene spread against per-gene mean: count variance (log–log, with Poisson and NB curves), SD of log2 counts, and SD of log2 array intensity.
    """)
    return


@app.cell
def _(
    COLORS,
    measure_array,
    measure_seq,
    n_rep_slider,
    np,
    phi_genes_slider,
    plt,
    sim_abundance,
):
    _rng = np.random.default_rng(7)
    _G, _n, _phi = 3000, n_rep_slider.value, phi_genes_slider.value
    _mu = 10 ** _rng.uniform(-1, 4, _G)
    _lam = sim_abundance(_rng, _mu[:, None], _phi, (_G, _n))
    _cnt = measure_seq(_rng, _lam)
    _arr = measure_array(_rng, _lam)

    _m, _v = _cnt.mean(1), _cnt.var(1, ddof=1)
    _keep = (_m > 0) & (_v > 0)
    _grid = np.logspace(-1, 4, 200)

    _fig, _ax = plt.subplots(1, 3, figsize=(14, 3.8))
    _ax[0].scatter(_m[_keep], _v[_keep], s=3, alpha=0.3, color=COLORS["seq"])
    _ax[0].plot(_grid, _grid, "k--", lw=1, label="Poisson: var = μ")
    _ax[0].plot(_grid, _grid + _phi * _grid**2, "k-", lw=1.5, label="NB: var = μ + φμ²")
    _ax[0].set(xscale="log", yscale="log", xlabel="mean count", ylabel="variance",
               title="RNA-seq counts")
    _ax[0].legend(frameon=False, fontsize=8)

    _l = np.log2(_cnt + 1)
    _ax[1].scatter(_l.mean(1), _l.std(1, ddof=1), s=3, alpha=0.3, color=COLORS["seq"])
    _ax[1].axhline(np.sqrt(_phi) / np.log(2), color="k", lw=1, ls=":",
                   label="high-count limit ≈ √φ / ln 2")
    _ax[1].set(xlabel="mean log2(count+1)", ylabel="SD", title="RNA-seq, log2(count+1)")
    _ax[1].legend(frameon=False, fontsize=8)

    _ax[2].scatter(_arr.mean(1), _arr.std(1, ddof=1), s=3, alpha=0.3, color=COLORS["array"])
    _ax[2].set(xlabel="mean log2 intensity", ylabel="SD", title="Microarray, log2 intensity")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What to notice**

    * **Counts:** variance is tied to the mean. Low-count genes hug the Poisson line
      (sampling noise dominates); high-count genes follow $\phi\mu^2$ (biology dominates).
      A test that assumes one shared $\sigma^2$ is wrong almost everywhere.
    * **log2(count+1):** the log *mostly* stabilizes variance for highly expressed genes,
      but low-count genes have a hump of extra noise and then collapse toward zero spread.
      This is the motivation for `voom` (model this trend) and for `DESeq2`/`edgeR`
      (model the counts directly).
    * **Array:** after the log, spread is roughly constant except where background or
      saturation compresses it. This is why `limma` (moderated t-tests on log
      intensities) worked so well for arrays.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1b. Real data: dendritic cells on arrays vs RNA-seq

    * **Microarray:** [GSE8658](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8658),
      human monocytes → monocyte-derived DCs (6 donors, several time points/treatments),
      Affymetrix HG-U133 Plus 2, GC-RMA summarized. 63 arrays. Probes collapsed to genes
      by taking the highest-expressed probe.
    * **RNA-seq:** monocyte-derived DC experiments from the lab (UVB/IFNβ, TNF/IL32) plus a
      public DC stimulation set. Raw counts, libraries < 1 M reads dropped.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: load the data.** `load_array` downloads the GSE8658 series matrix (cached in `data/array/`), log2-transforms the GC-RMA values, maps probes to genes and keeps the highest-expressed probe per gene. `load_rnaseq` reads the three DC raw-count tables, keeps genes present in all of them, and drops libraries under 1 M reads. The RNA-seq folder can be changed with the `BULK_RNASEQ_DIR` environment variable.
    """)
    return


@app.cell
def _(Path, mo, np, pd):
    import gzip
    import io
    import os
    import urllib.request

    DATA_DIR = Path(__file__).parent / "data"
    RNASEQ_DIR = Path(
        os.environ.get("BULK_RNASEQ_DIR", "~/Dropbox_umms/projects/carol_chemokine_data")
    ).expanduser()
    GSE8658_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8658/matrix/GSE8658_series_matrix.txt.gz"

    def load_array():
        """GSE8658 series matrix -> gene x sample log2 matrix + sample table."""
        path = DATA_DIR / "array" / "GSE8658_series_matrix.txt.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(GSE8658_URL, path)
        lines = gzip.open(path, "rt").read().splitlines()
        titles = next(l for l in lines if l.startswith("!Sample_title")).split("\t")[1:]
        titles = [t.strip('"') for t in titles]
        start = lines.index("!series_matrix_table_begin")
        expr = pd.read_csv(io.StringIO("\n".join(lines[start + 1 : -1])), sep="\t", index_col=0)
        expr.columns = titles
        expr = np.log2(expr)  # GC-RMA values are deposited on the linear scale

        symbol = pd.read_csv(DATA_DIR / "array" / "GPL570_probe2symbol.tsv.gz", sep="\t", index_col=0)[
            "Gene Symbol"
        ].reindex(expr.index)
        keep = symbol.notna() & ~symbol.str.contains("///", regex=False, na=True)
        best = expr[keep].mean(axis=1).groupby(symbol[keep]).idxmax()
        genes = expr.loc[best.values]
        genes.index = best.index

        def _group(t):
            if t.startswith("MC"):
                return "monocyte"
            time, *rest = t.split()
            return f"{time} DC" + (" untreated" if len(rest) == 1 and rest[0][:2] == "DC" and rest[0][2:].isdigit() else f" {' '.join(rest[1:]) or 'other'}")

        samples = pd.DataFrame({"group": [_group(t) for t in titles]}, index=titles)
        return genes, samples

    def load_rnaseq(min_lib=1e6):
        """Pool DC raw-count tables -> gene x sample counts + sample table."""
        tables = {
            "UVB": pd.read_csv(RNASEQ_DIR / "DCs_UVB/DC_uvb_meta_raw.tsv", sep="\t", index_col=0),
            "TNF_IL32": pd.read_csv(RNASEQ_DIR / "DCs_TNF_IL32/DC_TNF_IL32_raw.tsv", sep="\t").set_index("gene"),
            "public": pd.read_csv(RNASEQ_DIR / "DC_public_data/DC_pd_raw.tsv", sep="\t", index_col=0)
            .round()
            .astype(int)
            .add_prefix("pub_"),
        }
        counts = pd.concat(tables.values(), axis=1, join="inner")
        source = np.concatenate([[k] * t.shape[1] for k, t in tables.items()])
        samples = pd.DataFrame({"source": source, "lib_size": counts.sum().values}, index=counts.columns)
        # condition = sample name minus the donor / replicate tag
        samples["group"] = [
            f"{src}: " + (c.split("_", 1)[1] if src != "public" else c[4:].rsplit("_", 1)[0])
            for c, src in zip(counts.columns, source)
        ]
        keep = samples.lib_size >= min_lib
        return counts.loc[:, keep], samples[keep]

    arr_genes, arr_samples = load_array()
    if not RNASEQ_DIR.exists():
        mo.stop(True, mo.callout(mo.md(f"RNA-seq folder not found: `{RNASEQ_DIR}`. Set `BULK_RNASEQ_DIR`."), kind="warn"))
    seq_counts, seq_samples = load_rnaseq()
    seq_cpm = seq_counts / seq_counts.sum() * 1e6
    mo.md(
        f"Loaded **{arr_genes.shape[1]} arrays** × {arr_genes.shape[0]:,} genes and "
        f"**{seq_counts.shape[1]} RNA-seq libraries** × {seq_counts.shape[0]:,} genes "
        f"(library sizes {seq_samples.lib_size.min() / 1e6:.1f}–{seq_samples.lib_size.max() / 1e6:.0f} M reads)."
    )
    return arr_genes, arr_samples, seq_counts, seq_cpm, seq_samples


@app.cell
def _(mo):
    mo.md(r"""
    ### Mean–variance in replicate groups

    Pick one homogeneous group per technology (same cell state, different donors) so the
    spread is biological + technical noise, not treatment. Both are shown on the log2
    scale, which is how we would feed them to a linear model. The right panel puts the
    two trends on a common x-axis (expression percentile).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Choose one replicate group per technology (only groups with at least 4 samples are listed).
    """)
    return


@app.cell
def _(arr_samples, mo, seq_samples):
    _arr_groups = arr_samples.group.value_counts()
    _seq_groups = seq_samples.group.value_counts()
    arr_group = mo.ui.dropdown(
        {f"{g} (n={n})": g for g, n in _arr_groups[_arr_groups >= 4].items()},
        value=f"5d DC untreated (n={_arr_groups['5d DC untreated']})",
        label="array group",
    )
    seq_group = mo.ui.dropdown(
        {f"{g} (n={n})": g for g, n in _seq_groups[_seq_groups >= 4].items()},
        value=f"public: Ctrl (n={_seq_groups['public: Ctrl']})",
        label="RNA-seq group",
    )
    mo.hstack([arr_group, seq_group], justify="start", gap=2)
    return arr_group, seq_group


@app.cell
def _(mo):
    mo.md(r"""
    **Code: real mean–variance plots.** For each technology, compute every gene's mean and SD across the replicate group on the log2 scale (RNA-seq counts are first scaled to a common library size). A lowess curve shows the trend. Panel 3 shows RNA-seq on the count scale against the Poisson line. Panel 4 overlays both trends against expression percentile so the two technologies share an x-axis.
    """)
    return


@app.cell
def _(
    COLORS,
    arr_genes,
    arr_group,
    arr_samples,
    np,
    plt,
    seq_counts,
    seq_group,
    seq_samples,
    sm,
):
    _lowess = sm.nonparametric.lowess

    _A = arr_genes.loc[:, arr_samples.group == arr_group.value]
    _S = seq_counts.loc[:, seq_samples.group == seq_group.value]
    _S = _S[(_S > 0).any(axis=1)]
    _norm = _S / _S.sum() * _S.sum().mean()  # simple library-size scaling
    _L = np.log2(_norm + 1)

    _a_m, _a_sd = _A.mean(axis=1), _A.std(axis=1)
    _s_m, _s_sd = _L.mean(axis=1), _L.std(axis=1)
    _a_fit = _lowess(_a_sd, _a_m, frac=0.2, return_sorted=True)
    _s_fit = _lowess(_s_sd, _s_m, frac=0.2, return_sorted=True)

    _fig, _ax = plt.subplots(1, 4, figsize=(17, 3.9))
    _ax[0].scatter(_a_m, _a_sd, s=2, alpha=0.15, color=COLORS["array"])
    _ax[0].plot(*_a_fit.T, "k-", lw=2)
    _ax[0].set(xlabel="mean log2 intensity", ylabel="SD (log2)", title=f"Array: {arr_group.value} (n={_A.shape[1]})")
    _ax[1].scatter(_s_m, _s_sd, s=2, alpha=0.15, color=COLORS["seq"])
    _ax[1].plot(*_s_fit.T, "k-", lw=2)
    _ax[1].set(xlabel="mean log2(norm. count + 1)", ylabel="SD (log2)", title=f"RNA-seq: {seq_group.value} (n={_S.shape[1]})")
    _ymax = np.quantile(np.r_[_a_sd, _s_sd], 0.999)
    _ax[0].set_ylim(0, _ymax)
    _ax[1].set_ylim(0, _ymax)

    _nm, _nv = _norm.mean(axis=1), _norm.var(axis=1)
    _ok = (_nm > 0) & (_nv > 0)
    _ax[2].scatter(_nm[_ok], _nv[_ok], s=2, alpha=0.15, color=COLORS["seq"])
    _g = np.logspace(np.log10(_nm[_ok].min()), np.log10(_nm.max()), 100)
    _ax[2].plot(_g, _g, "k--", lw=1, label="Poisson: var = mean")
    _ax[2].set(xscale="log", yscale="log", xlabel="mean count", ylabel="variance", title="RNA-seq, count scale")
    _ax[2].legend(frameon=False, fontsize=8)

    for _m, _fit, _c, _lab in ((_a_m, _a_fit, COLORS["array"], "array"), (_s_m, _s_fit, COLORS["seq"], "RNA-seq")):
        _pct = np.searchsorted(np.sort(_m.values), _fit[:, 0]) / len(_m) * 100
        _ax[3].plot(_pct, _fit[:, 1], color=_c, lw=2.5, label=_lab)
    _ax[3].set(xlabel="expression percentile (within technology)", ylabel="SD trend (log2)",
               title="Trends on a common axis", ylim=(0, None))
    _ax[3].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the real data (be honest with the class)**

    * **Both** technologies have a mean–SD trend on the log scale. Arrays aren't perfectly
      homoscedastic: GC-RMA squeezes background probes toward a floor (SD → 0 at the low
      end) and mid-range probes are noisiest.
    * The RNA-seq trend is **steeper and has a known cause**: at low counts Poisson sampling
      noise dominates (the points sitting on the dashed line in panel 3), so the log-scale SD
      climbs as expression drops, until genes have so many zeros that the SD collapses.
      This variance is a *function of the mean*, which is exactly what a NB GLM encodes.
    * For arrays, `limma` with an intensity trend (`eBayes(trend=TRUE)`) is enough.
      For RNA-seq you can either model the counts (DESeq2/edgeR GLMs) or estimate this
      trend and pass it as precision weights to a linear model (`limma-voom`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### One gene across many samples

    Now pool *all* samples (every condition, every donor) and look at a single
    housekeeping gene. Pick a highly expressed one (GAPDH, ACTB, B2M) and a low one (TBP,
    GUSB, HPRT1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Pick a housekeeping gene from the list, or type any gene symbol present in both datasets.
    """)
    return


@app.cell
def _(arr_genes, mo, seq_counts):
    _hk = ["GAPDH", "ACTB", "B2M", "PPIA", "RPLP0", "HPRT1", "GUSB", "TBP", "HMBS", "POLR2A"]
    _common = sorted(set(arr_genes.index) & set(seq_counts.index))
    hk_gene = mo.ui.dropdown([g for g in _hk if g in _common], value="GAPDH", label="housekeeping gene")
    any_gene = mo.ui.text(placeholder="or type any gene symbol", label="")
    mo.hstack([hk_gene, any_gene], justify="start", gap=2)
    return any_gene, hk_gene


@app.cell
def _(mo):
    mo.md(r"""
    **Code: one gene across all samples.** Histogram of the gene in all arrays (log2), in all RNA-seq libraries as raw counts (stacked by experiment), and as log2 CPM. The bottom row has normal QQ plots with a Shapiro–Wilk p-value. The table shows how strongly the raw count tracks library size.
    """)
    return


@app.cell
def _(
    COLORS,
    any_gene,
    arr_genes,
    hk_gene,
    mo,
    np,
    pd,
    plt,
    seq_counts,
    seq_cpm,
    seq_samples,
    stats,
):
    _gene = any_gene.value.strip().upper() or hk_gene.value
    if _gene not in arr_genes.index or _gene not in seq_counts.index:
        mo.stop(True, mo.callout(mo.md(f"`{_gene}` is not in both datasets."), kind="warn"))

    _a = arr_genes.loc[_gene]
    _raw = seq_counts.loc[_gene]
    _lcpm = np.log2(seq_cpm.loc[_gene] + 1)

    _fig, _ax = plt.subplots(2, 3, figsize=(15, 6.5), gridspec_kw={"height_ratios": [1.3, 1]})
    _ax[0, 0].hist(_a, bins=20, color=COLORS["array"])
    _ax[0, 0].set(title=f"Array {_gene}: log2 intensity (n={len(_a)})", xlabel="log2 intensity")
    _src_colors = {"UVB": "#C44E52", "TNF_IL32": "#8172B3", "public": "#937860"}
    _bins = np.histogram_bin_edges(_raw, bins=25)
    _ax[0, 1].hist([_raw[seq_samples.source == s] for s in _src_colors], bins=_bins, stacked=True,
                   color=list(_src_colors.values()), label=list(_src_colors))
    _ax[0, 1].set(title=f"RNA-seq {_gene}: raw counts (n={len(_raw)})", xlabel="reads")
    _ax[0, 1].legend(frameon=False, fontsize=8, title="experiment", title_fontsize=8)
    _ax[0, 2].hist(_lcpm, bins=20, color=COLORS["seq"])
    _ax[0, 2].set(title=f"RNA-seq {_gene}: log2(CPM + 1)", xlabel="log2 CPM")

    for _j, (_v, _c, _lab) in enumerate(((_a, COLORS["array"], "array log2"), (_raw, COLORS["seq"], "raw counts"), (_lcpm, COLORS["seq"], "log2 CPM"))):
        stats.probplot(_v, dist="norm", plot=_ax[1, _j])
        _ax[1, _j].get_lines()[0].set(markerfacecolor=_c, markeredgecolor=_c, markersize=4)
        _ax[1, _j].set(title=f"normal QQ: {_lab}  (Shapiro p = {stats.shapiro(_v).pvalue:.2g})")
    for _a_ in _ax[0]:
        _a_.set_yticks([])
    _fig.tight_layout()

    _tab = pd.DataFrame(
        {
            "median raw count": [int(_raw.median())],
            "min raw count": [int(_raw.min())],
            "zeros": [int((_raw == 0).sum())],
            "corr(raw count, library size)": [round(np.corrcoef(_raw, seq_samples.lib_size)[0, 1], 2)],
        }
    )
    mo.vstack([_fig, mo.ui.table(_tab, selection=None, show_download=False)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What this shows (and doesn't)**

    * **Array, GAPDH/ACTB:** one bell-shaped log2 intensity. Ready for a linear model.
    * **RNA-seq raw counts:** clumped by experiment, because raw counts mostly track
      **sequencing depth** (see the correlation with library size). Counts from samples
      of different depth are not comparable. The GLM handles this with the offset
      $\log s_j$ instead of dividing the data.
    * **After CPM + log2, GAPDH looks fairly normal too.** High-count genes are in the
      regime where NB ≈ log-normal, so for them the distinction is small.
    * The difference shows up for **low-count genes**. Try TBP or GUSB, or type a
      chemokine like `CXCL10` or `CCL19`: discreteness, zeros, and a variance set by the
      count level. Most genes in a typical RNA-seq experiment are in this regime.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The two-sample t-test *is* a linear model

    Take one gene, two groups (A = control, B = treated), $n$ samples each, log-scale
    expression $y_i$. Encode group with an indicator $x_i$ ($0$ for A, $1$ for B):

    $$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)$$

    * $\beta_0$ = mean of group A
    * $\beta_0 + \beta_1$ = mean of group B, so $\beta_1$ = **difference in means** (the log fold change)
    * Least squares gives $\hat\beta_1 = \bar y_B - \bar y_A$
    * $t = \hat\beta_1 / \text{SE}(\hat\beta_1)$ with $2n - 2$ degrees of freedom

    That is *exactly* Student's (equal-variance) t-test. Testing "the means differ" is
    testing $H_0: \beta_1 = 0$. In matrix form $\mathbf{y} = X\boldsymbol\beta + \boldsymbol\varepsilon$,
    where $X$ is the **design matrix**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Sample size per group, the true difference between groups (β₁), the noise SD, and a random seed to draw a new data set.
    """)
    return


@app.cell
def _(mo):
    n_grp = mo.ui.slider(2, 20, value=4, label="n per group", show_value=True)
    delta = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="true log2FC (β₁)", show_value=True)
    sigma = mo.ui.slider(0.1, 2.0, step=0.05, value=0.5, label="noise SD σ", show_value=True)
    seed = mo.ui.number(1, 10_000, value=1, label="seed")
    mo.hstack([n_grp, delta, sigma, seed], justify="start", gap=2)
    return delta, n_grp, seed, sigma


@app.cell
def _(mo):
    mo.md(r"""
    **Code: simulate one gene.** Draw *n* log-expression values per group from normal distributions whose means differ by the chosen log2FC. Store them in a table with an explicit 0/1 column `x`, which becomes the design-matrix column.
    """)
    return


@app.cell
def _(delta, n_grp, np, pd, seed, sigma):
    _rng = np.random.default_rng(seed.value)
    _n = n_grp.value
    tt_df = pd.DataFrame(
        {
            "group": ["A"] * _n + ["B"] * _n,
            "x": [0] * _n + [1] * _n,
            "y": np.r_[
                _rng.normal(8.0, sigma.value, _n),
                _rng.normal(8.0 + delta.value, sigma.value, _n),
            ],
        }
    )
    return (tt_df,)


@app.cell
def _(mo):
    mo.md(r"""
    **Code: t-test vs linear model.** Run Student's t-test (`scipy.stats.ttest_ind`, `equal_var=True`) and fit OLS of y on [1, x] (`statsmodels`). Plot the data with the fitted line, whose slope is β₁, and show both results side by side with the design matrix.
    """)
    return


@app.cell
def _(COLORS, mo, np, pd, plt, sm, stats, tt_df):
    _yA = tt_df.loc[tt_df.x == 0, "y"]
    _yB = tt_df.loc[tt_df.x == 1, "y"]

    _tt = stats.ttest_ind(_yB, _yA, equal_var=True)
    _X = sm.add_constant(tt_df[["x"]])
    _ols = sm.OLS(tt_df["y"], _X).fit()
    _b0, _b1 = _ols.params["const"], _ols.params["x"]

    _rng = np.random.default_rng(0)
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _g, _x in [("A", 0), ("B", 1)]:
        _yy = tt_df.loc[tt_df.x == _x, "y"]
        _ax.scatter(_x + _rng.uniform(-0.06, 0.06, len(_yy)), _yy, color=COLORS[_g], zorder=3)
        _ax.hlines(_yy.mean(), _x - 0.15, _x + 0.15, color=COLORS[_g], lw=2)
    _ax.plot([0, 1], [_b0, _b0 + _b1], "k-", lw=1.5, label=f"fit: y = {_b0:.2f} + {_b1:.2f}·x")
    _ax.annotate("", xy=(1.25, _b0 + _b1), xytext=(1.25, _b0),
                 arrowprops=dict(arrowstyle="<->", color="gray"))
    _ax.axhline(_b0, color="gray", lw=0.5, ls=":")
    _ax.text(1.3, _b0 + _b1 / 2, "β₁", va="center", color="gray")
    _ax.set(xticks=[0, 1], xticklabels=["A (x=0)", "B (x=1)"], xlim=(-0.4, 1.5),
            ylabel="log2 expression")
    _ax.legend(frameon=False, loc="upper left", fontsize=8)
    _fig.tight_layout()

    _cmp = pd.DataFrame(
        {
            "t-test (scipy)": [_yB.mean() - _yA.mean(), np.nan, _tt.statistic, _tt.df, _tt.pvalue],
            "linear model (OLS), β₁": [_b1, _ols.bse["x"], _ols.tvalues["x"], _ols.df_resid, _ols.pvalues["x"]],
        },
        index=["difference / β₁", "SE", "t", "df", "p-value"],
    )

    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    mo.md("**Same numbers, two vocabularies**"),
                    mo.ui.table(_cmp.reset_index(names=""), selection=None, show_download=False),
                    mo.md("**Design matrix X** (what the model actually sees)"),
                    mo.ui.table(_X.assign(y=tt_df["y"].round(3), group=tt_df["group"]),
                                selection=None, page_size=8, show_download=False),
                ]
            ),
        ],
        widths=[1, 1.2],
        align="start",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Note: scipy's default `ttest_ind(..., equal_var=False)` is **Welch's** test, which
    allows different variances per group. That's not an ordinary linear model (it's a
    weighted one). The classical Student's t-test assumes one $\sigma^2$, as OLS does.

    In R the same equivalence is:

    ```r
    t.test(y ~ group, var.equal = TRUE)
    summary(lm(y ~ group))   # same t and p for the 'groupB' coefficient
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why bother rewriting a t-test as a model? Because experiments have structure.

    Suppose half the samples were processed on Monday and half on Friday (batch).
    The t-test has nowhere to put that. The model just gets another column:

    $$y_i = \beta_0 + \beta_1\,\text{treated}_i + \beta_2\,\text{batch}_i + \varepsilon_i$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Batch effect size, whether batch is balanced across groups or partially confounded with treatment, group size, true log2FC, and seed.
    """)
    return


@app.cell
def _(mo):
    batch_eff = mo.ui.slider(0.0, 3.0, step=0.1, value=1.5, label="batch effect (log2)", show_value=True)
    design = mo.ui.dropdown(
        {"balanced": 0.5, "partially confounded": 0.8},
        value="balanced",
        label="design",
    )
    n_b = mo.ui.slider(4, 20, step=2, value=6, label="n per group", show_value=True)
    lfc_b = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="true log2FC", show_value=True)
    seed_b = mo.ui.number(1, 10_000, value=3, label="seed")
    mo.hstack([batch_eff, design, n_b, lfc_b, seed_b], justify="start", gap=1.5)
    return batch_eff, design, lfc_b, n_b, seed_b


@app.cell
def _(mo):
    mo.md(r"""
    **Code: t-test vs model with batch.** Simulate expression = baseline + treatment effect + batch effect + noise. Compare the t-test (ignores batch) with `ols('y ~ treated + C(batch)')` on the estimated log2FC and p-value.
    """)
    return


@app.cell
def _(
    COLORS,
    batch_eff,
    design,
    lfc_b,
    mo,
    n_b,
    np,
    pd,
    plt,
    seed_b,
    smf,
    stats,
):
    _rng = np.random.default_rng(seed_b.value)
    _n = n_b.value
    _frac = design.value  # fraction of B samples in batch 2 (and of A samples in batch 1)
    _kB = int(round(_frac * _n))
    # A: _kB samples in batch 1 (=0), rest in batch 2; B: the mirror image
    _batch = np.r_[np.where(np.arange(_n) < _kB, 0, 1), np.where(np.arange(_n) < _kB, 1, 0)]
    _x = np.r_[np.zeros(_n), np.ones(_n)]
    _y = 8 + lfc_b.value * _x + batch_eff.value * _batch + _rng.normal(0, 0.4, 2 * _n)
    _d = pd.DataFrame({"y": _y, "treated": _x.astype(int), "batch": _batch})

    _tt = stats.ttest_ind(_y[_x == 1], _y[_x == 0], equal_var=True)
    _m = smf.ols("y ~ treated + C(batch)", data=_d).fit()

    _fig, _ax = plt.subplots(figsize=(5, 3.8))
    _mk = {0: "o", 1: "s"}
    for _b in (0, 1):
        for _g, _xx in (("A", 0), ("B", 1)):
            _s = (_d.batch == _b) & (_d.treated == _xx)
            _ax.scatter(_xx + (_b - 0.5) * 0.25 + _rng.uniform(-0.04, 0.04, _s.sum()), _d.y[_s],
                        color=COLORS[_g], marker=_mk[_b], edgecolor="k", lw=0.4,
                        label=f"batch {_b + 1}" if _g == "A" else None)
    _ax.set(xticks=[0, 1], xticklabels=["A", "B"], ylabel="log2 expression")
    _ax.legend(frameon=False, fontsize=8, title="marker = batch", title_fontsize=8)
    _fig.tight_layout()

    _res = pd.DataFrame(
        {
            "estimated log2FC": [_y[_x == 1].mean() - _y[_x == 0].mean(), _m.params["treated"]],
            "p-value": [_tt.pvalue, _m.pvalues["treated"]],
        },
        index=["t-test  (y ~ treated)", "model  (y ~ treated + batch)"],
    ).round(4)

    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    mo.md(f"True log2FC = **{lfc_b.value}**"),
                    mo.ui.table(_res.reset_index(names="method"), selection=None, show_download=False),
                    mo.md(
                        """
    * **Balanced:** batch doesn't bias the t-test, but its variance gets dumped into
      the error term → bigger SE, bigger p-value. The model removes it → more power.
    * **Partially confounded:** most B samples were in batch 2. Now the t-test's
      estimate is **biased** (it absorbs part of the batch effect). The model separates them.
    * If design were *fully* confounded no method could separate them. Design matters!
    """
                    ),
                ]
            ),
        ],
        widths=[1, 1.3],
        align="start",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The same idea extends naturally:

    | Experimental question | t-test world | Model world |
    |---|---|---|
    | 2 groups | t-test | `y ~ group` |
    | 3+ groups | ANOVA | `y ~ group` (more indicator columns) |
    | Adjust for batch / sex / RIN | ✗ | `y ~ batch + sex + group` |
    | Paired (before/after, same donor) | paired t-test | `y ~ donor + treatment` or mixed model `y ~ treatment + (1 | donor)` |
    | Interaction (does treatment effect differ by genotype?) | ✗ | `y ~ genotype * treatment` |
    | Counts, not Gaussian | ✗ | **GLM**: change the distribution + link |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. From linear model to GLM for counts (preview)

    Keep the *same* linear predictor, change two things:

    $$y_{ij} \sim \text{NB}(\mu_{ij}, \phi_i), \qquad \log \mu_{ij} = \beta_0 + \beta_1 x_j \;(+\log s_j)$$

    * **Distribution:** negative binomial instead of normal, so $\operatorname{Var} = \mu + \phi\mu^2$
      (what we saw in Part 1).
    * **Link:** model $\log\mu$, so $\beta_1$ is a log fold change and fitted means stay positive.
    * $\log s_j$ is an *offset* for library size / size factor. No need to pre-normalize counts.

    This is the model inside **DESeq2** and **edgeR**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: controls.** Baseline mean count, true log2FC, dispersion and group size for the count simulations below.
    """)
    return


@app.cell
def _(mo):
    base_mu = mo.ui.slider(
        steps=[1, 2, 5, 10, 20, 50, 100, 500, 1000], value=10,
        label="baseline mean count", show_value=True,
    )
    lfc_g = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="true log2FC", show_value=True)
    phi_g = mo.ui.slider(0.01, 0.5, step=0.01, value=0.1, label="dispersion φ", show_value=True)
    n_g = mo.ui.slider(2, 12, value=3, label="n per group", show_value=True)
    mo.hstack([base_mu, lfc_g, phi_g, n_g], justify="start", gap=1.5)
    return base_mu, lfc_g, n_g, phi_g


@app.cell
def _(mo):
    mo.md(r"""
    **Code: one gene, two models.** Simulate NB counts for two groups. Fit (1) OLS on log2(count+1), which is the t-test, and (2) a negative-binomial GLM with log link and the true φ (`sm.GLM(..., family=NegativeBinomial)`). Compare the estimated log2FCs and p-values.
    """)
    return


@app.cell
def _(COLORS, base_mu, lfc_g, mo, n_g, np, phi_g, plt, sim_abundance, sm):
    _rng = np.random.default_rng(11)
    _n, _phi = n_g.value, phi_g.value
    _muA, _muB = base_mu.value, base_mu.value * 2**lfc_g.value
    _yA = _rng.poisson(sim_abundance(_rng, _muA, _phi, _n))
    _yB = _rng.poisson(sim_abundance(_rng, _muB, _phi, _n))
    _y = np.r_[_yA, _yB]
    _X = sm.add_constant(np.r_[np.zeros(_n), np.ones(_n)])

    _glm = sm.GLM(_y, _X, family=sm.families.NegativeBinomial(alpha=_phi)).fit()
    _lm = sm.OLS(np.log2(_y + 1), _X).fit()

    _fig, _ax = plt.subplots(figsize=(5, 3.8))
    for _g, _x, _yy in (("A", 0, _yA), ("B", 1, _yB)):
        _ax.scatter(np.full(_n, _x) + _rng.uniform(-0.05, 0.05, _n), _yy, color=COLORS[_g], zorder=3)
    _fit = np.exp(_glm.params[0] + _glm.params[1] * np.array([0, 1]))
    _ax.plot([0, 1], _fit, "k-o", ms=4, label="NB GLM fitted means")
    _ax.plot([0, 1], [_muA, _muB], "--", color="gray", label="true means")
    _ax.set(xticks=[0, 1], xticklabels=["A", "B"], ylabel="raw counts", yscale="symlog")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()

    mo.hstack(
        [
            _fig,
            mo.md(
                f"""
    | | log2FC estimate | p-value |
    |---|---|---|
    | **truth** | {lfc_g.value:.2f} | |
    | OLS on log2(count+1) (= t-test) | {_lm.params[1]:.2f} | {_lm.pvalues[1]:.3g} |
    | NB GLM, log link | {_glm.params[1] / np.log(2):.2f} | {_glm.pvalues[1]:.3g} |

    The GLM coefficient is on the natural-log scale; divide by ln 2 for log2FC.

    At low counts the pseudo-count in log2(count+1) **shrinks the fold change toward 0**.
    The GLM estimates it on the count scale directly.

    *(Here the GLM is given the true φ. Real tools must estimate φ from few samples —
    see below.)*
    """
            ),
        ],
        widths=[1, 1.2],
        align="start",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Many genes: false positives and power

    Simulate 4000 genes at the chosen baseline mean. Under the **null** (log2FC = 0) a
    calibrated test should give ~5% of p-values below 0.05. Under the **alternative**,
    the fraction below 0.05 is the power.

    For a two-group design the NB GLM has a closed form: $\hat\mu_g = \bar y_g$ and
    $\operatorname{SE}(\hat\beta_1)^2 = \sum_g \big[n_g \hat\mu_g / (1+\phi\hat\mu_g)\big]^{-1}$,
    so we can do this fast without calling a fitting routine 4000 times.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Code: many-gene simulation.** Simulate 4000 null genes and 4000 truly changed genes. Test each with a t-test on log2(count+1) and with the closed-form NB Wald test, then report the fraction with p < 0.05.
    """)
    return


@app.cell
def _(base_mu, lfc_g, mo, n_g, np, pd, phi_g, sim_abundance, stats):
    def _nb_wald(a, b, phi):
        n = a.shape[1]
        ma, mb = a.mean(1), b.mean(1)
        ok = (ma > 0) & (mb > 0)
        safe_a, safe_b = np.where(ok, ma, 1.0), np.where(ok, mb, 1.0)
        beta = np.log(safe_b / safe_a)
        se = np.sqrt((1 + phi * safe_a) / (n * safe_a) + (1 + phi * safe_b) / (n * safe_b))
        return np.where(ok, 2 * stats.norm.sf(np.abs(beta / se)), 1.0)

    def _run(lfc):
        rng = np.random.default_rng(2024)
        G, n, phi, mu = 4000, n_g.value, phi_g.value, base_mu.value
        a = rng.poisson(sim_abundance(rng, mu, phi, (G, n)))
        b = rng.poisson(sim_abundance(rng, mu * 2**lfc, phi, (G, n)))
        with np.errstate(all="ignore"):
            p_t = stats.ttest_ind(np.log2(b + 1), np.log2(a + 1), axis=1).pvalue
        p_t = np.nan_to_num(p_t, nan=1.0)  # genes with zero variance in both groups
        return (p_t < 0.05).mean(), (_nb_wald(a, b, phi) < 0.05).mean()

    _null = _run(0.0)
    _alt = _run(lfc_g.value)
    _tab = pd.DataFrame(
        {
            "false-positive rate (log2FC=0)": [_null[0], _null[1]],
            f"power (log2FC={lfc_g.value})": [_alt[0], _alt[1]],
        },
        index=["t-test on log2(count+1)", "NB GLM Wald (φ known)"],
    ).map(lambda v: f"{100 * v:.1f}%")
    mo.ui.table(_tab.reset_index(names="test"), selection=None, show_download=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Honest reading of this table**

    * The t-test on log counts is reasonably well *calibrated* (≈5% false positives) at
      moderate counts. The t-test is robust. Its problem is **power**: with
      n = 3 it must estimate a variance from 4 degrees of freedom, *per gene*.
    * The GLM here is an **oracle**: it is told the true φ. Its power advantage is mostly
      knowing the variance structure. Real data never hand you φ.
    * That's the bridge to the next part: **DESeq2/edgeR estimate φ by borrowing
      information across thousands of genes** (dispersion trend + empirical-Bayes
      shrinkage), which gets you close to the oracle. `limma`'s moderated t does the same
      for σ² on the log scale.
    * Try base mean = 1–2: zeros dominate, everything struggles, and the Wald test
      becomes conservative. Low counts carry little information whatever you do.

    ### Next steps (to build)

    * Estimate φ per gene with n = 3 → show how noisy it is → shrinkage.
    * Library size differences: t-test on unnormalized counts vs GLM with offset.
    * Paired / repeated-measures designs → mixed models (`dream`, `glmmTMB`).
    * Swap the simulation for a real dataset (e.g. `airway`).
    """)
    return


if __name__ == "__main__":
    app.run()
