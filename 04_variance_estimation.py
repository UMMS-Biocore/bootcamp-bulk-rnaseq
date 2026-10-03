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
    # 4. Estimating each gene's variance

    Every DE test divides an effect by its uncertainty, so it needs each gene's variance.
    This notebook makes three points:

    1. With a handful of replicates, a gene's variance **cannot be estimated from that
       gene's own data**.
    2. In RNA-seq the variance is **strongly tied to the expression level**; on arrays
       much less so.
    3. So we **fit a curve of variance vs mean across all genes** and use it to give each
       gene a better variance estimate. That is what DESeq2, edgeR and limma (eBayes,
       voom) all do in some form.

    The first cell loads the libraries.
    """)
    return


@app.cell
def _():
    import gzip
    import io
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import optimize, special, stats

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    ARRAY_COLOR, SEQ_COLOR = "#55A868", "#C44E52"
    DATA_DIR = Path(__file__).parent / "data"
    RNASEQ_DIR = Path(os.environ.get("BULK_RNASEQ_DIR", "~/Dropbox_umms/projects/carol_chemokine_data")).expanduser()
    return (
        ARRAY_COLOR,
        DATA_DIR,
        RNASEQ_DIR,
        SEQ_COLOR,
        gzip,
        io,
        mo,
        np,
        optimize,
        pd,
        plt,
        sm,
        special,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How noisy is one gene's variance estimate?

    If a gene's values are normal with true variance $\sigma^2$, the sample variance from
    $n$ replicates follows

    $$\frac{s^2}{\sigma^2} \sim \frac{\chi^2_{n-1}}{n-1}$$

    This holds no matter how well behaved the gene is. Move the slider to see the
    range of $s^2/\sigma^2$ you get from $n$ replicates. The table gives the 95% range
    for several $n$.
    """)
    return


@app.cell
def _(mo):
    n_reps = mo.ui.slider(2, 100, value=3, label="replicates n", show_value=True)
    n_reps
    return (n_reps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot draws the distribution of the estimated variance divided by the true
    variance (1 = perfect). The shaded band is the central 95%.
    """)
    return


@app.cell
def _(mo, n_reps, np, pd, plt, stats):
    _n = n_reps.value
    _dist = stats.chi2(_n - 1, scale=1 / (_n - 1))
    _lo, _hi = _dist.ppf([0.025, 0.975])
    _x = np.logspace(-3, 1.3, 600)
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.plot(_x, _dist.pdf(_x) * _x, color="k")  # density on a log x-axis
    _band = (_x >= _lo) & (_x <= _hi)
    _ax.fill_between(_x[_band], 0, (_dist.pdf(_x) * _x)[_band], color="#4C72B0", alpha=0.3)
    _ax.axvline(1, color="gray", ls="--")
    _ax.set(xscale="log", xlabel="estimated variance / true variance", yticks=[],
            title=f"n = {_n}: 95% of estimates fall between {_lo:.3f}× and {_hi:.2f}× the truth")

    _rows = []
    for _k in (2, 3, 4, 6, 10, 20, 50, 100):
        _a, _b = stats.chi2.ppf([0.025, 0.975], _k - 1) / (_k - 1)
        _rows.append({"n": _k, "2.5%": round(_a, 3), "97.5%": round(_b, 2), "fold range": round(_b / _a, 1)})
    mo.hstack([_fig, mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False)], widths=[2, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With 3 replicates the estimate spans a ~150-fold range. Some genes will look almost
    noise-free and give huge t-statistics (false positives); others will look very noisy
    and be missed. Even 10 replicates leave a 7-fold range.

    ## Real data

    One homogeneous group per technology, 6 donors each:

    * **RNA-seq:** public DC controls (~27 M reads), depth-normalized integer counts
      (size factor = library size / median library size).
    * **Array:** GSE8658 day-5 untreated DCs, log2 intensity. Genes above the 30th
      percentile of mean intensity (drops background probes).

    We split the 6 donors into two halves of 3. The first half is used to **estimate**
    variances; the second half is held out to **check** the estimates. Genes with mean
    count < 1 in either half are dropped.
    """)
    return


@app.cell
def _(DATA_DIR, RNASEQ_DIR, gzip, io, mo, np, pd):
    _pub = RNASEQ_DIR / "DC_public_data/DC_pd_raw.tsv"
    mo.stop(not _pub.exists(), mo.callout(mo.md(f"Not found: `{_pub}`. Set `BULK_RNASEQ_DIR`."), kind="warn"))
    _raw = pd.read_csv(_pub, sep="\t", index_col=0).round().astype(int)
    _raw = _raw[[c for c in _raw.columns if c.startswith("Ctrl")]]
    seq = np.rint(_raw / (_raw.sum() / _raw.sum().median())).astype(int)
    seq = seq[(seq.iloc[:, :3].mean(axis=1) >= 1) & (seq.iloc[:, 3:].mean(axis=1) >= 1)]

    _lines = gzip.open(DATA_DIR / "array" / "GSE8658_series_matrix.txt.gz", "rt").read().splitlines()
    _titles = [t.strip('"') for t in next(l for l in _lines if l.startswith("!Sample_title")).split("\t")[1:]]
    _start = _lines.index("!series_matrix_table_begin")
    _probes = np.log2(pd.read_csv(io.StringIO("\n".join(_lines[_start + 1 : -1])), sep="\t", index_col=0))
    _probes.columns = _titles
    _sym = pd.read_csv(DATA_DIR / "array" / "GPL570_probe2symbol.tsv.gz", sep="\t", index_col=0)["Gene Symbol"]
    _sym = _sym.reindex(_probes.index)
    _ok = _sym.notna() & ~_sym.str.contains("///", regex=False, na=True)
    _best = _probes[_ok].mean(axis=1).groupby(_sym[_ok]).idxmax()
    arr = _probes.loc[_best.values].set_axis(_best.index)
    arr = arr[[t for t in _titles if t.startswith("5d DC") and len(t.split()) == 2]]
    arr = arr[arr.mean(axis=1) > arr.mean(axis=1).quantile(0.3)]

    mo.md(f"RNA-seq: **{seq.shape[0]:,} genes** × {seq.shape[1]} samples. "
          f"Array: **{arr.shape[0]:,} genes** × {arr.shape[1]} samples.")
    return arr, seq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First the problem in real data: each gene's variance estimated from donors 1–3
    against the same gene's variance from donors 4–6. If one gene's estimate were
    reliable, the points would sit on the diagonal.
    """)
    return


@app.cell
def _(ARRAY_COLOR, SEQ_COLOR, arr, np, plt, seq):
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for _a, _X, _c, _t, _u in ((_ax[0], arr, ARRAY_COLOR, "Array", "log2 intensity²"),
                               (_ax[1], seq, SEQ_COLOR, "RNA-seq", "counts²")):
        _v1, _v2 = _X.iloc[:, :3].var(axis=1), _X.iloc[:, 3:].var(axis=1)
        _k = (_v1 > 0) & (_v2 > 0)
        _a.scatter(_v1[_k], _v2[_k], s=2, alpha=0.15, color=_c)
        _lim = [min(_v1[_k].min(), _v2[_k].min()), max(_v1[_k].max(), _v2[_k].max())]
        _a.plot(_lim, _lim, "k--", lw=1)
        _r = np.corrcoef(np.log(_v1[_k]), np.log(_v2[_k]))[0, 1]
        _a.set(xscale="log", yscale="log", xlabel=f"variance, donors 1–3 ({_u})",
               ylabel="variance, donors 4–6", title=f"{_t}: correlation of log variances r = {_r:.2f}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **Array:** the two halves barely agree (r ≈ 0.3). A gene's variance from 3 donors
      is mostly noise.
    * **RNA-seq:** they agree much better (r ≈ 0.9), but **not because each gene's
      estimate is good**. Genes span four orders of magnitude in mean count, and the
      variance follows the mean. Both halves "know" the mean. The next section separates
      the two.

    ## 2. Variance vs expression: fit a curve across all genes

    Plot each gene's variance (donors 1–3) against its mean and fit a smooth curve
    (lowess on the log variance) through all genes at once. RNA-seq is shown on the count
    scale with log–log axes; the dashed line is the NB curve
    $\mu + \phi\mu^2$, the parametric form DESeq2 and edgeR use. Because the log of a
    noisy variance estimate is biased low on average, the fitted curve is corrected
    by the known chi-square offset.

    $R^2$ is the share of gene-to-gene variation in log variance explained by the mean.
    The next cell computes means, variances and the curve for both technologies.
    """)
    return


@app.cell
def _(arr, np, pd, seq, sm, special):
    DF = 2  # residual degrees of freedom with 3 replicates


    def fit_trend(X, log_mean):
        """Per-gene mean and variance from donors 1–3, plus a lowess trend of log variance on mean."""
        half = X.iloc[:, :3]
        m, v = half.mean(axis=1), half.var(axis=1)
        held = X.iloc[:, 3:].var(axis=1)
        keep = (v > 0) & (held > 0)
        m, v, held = m[keep], v[keep], held[keep]
        x = np.log(m) if log_mean else m
        fit = sm.nonparametric.lowess(np.log(v), x, frac=0.3, return_sorted=False)
        r2 = 1 - np.var(np.log(v) - fit) / np.var(np.log(v))
        # E[log s^2] = log sigma^2 + digamma(d/2) - log(d/2): undo that bias
        trend = pd.Series(np.exp(fit - (special.digamma(DF / 2) - np.log(DF / 2))), index=m.index)
        return {"mean": m, "var": v, "held_out": held, "trend": trend, "r2": r2}


    fits = {"Array": fit_trend(arr, log_mean=False), "RNA-seq": fit_trend(seq, log_mean=True)}
    return DF, fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each gene's variance (donors 1–3) against its mean, with the fitted curve.
    """)
    return


@app.cell
def _(ARRAY_COLOR, SEQ_COLOR, fits, np, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for _a, (_name, _f), _c in zip(_ax, fits.items(), (ARRAY_COLOR, SEQ_COLOR)):
        _o = np.argsort(_f["mean"].values)
        _a.scatter(_f["mean"], _f["var"], s=2, alpha=0.12, color=_c)
        _a.plot(_f["mean"].values[_o], _f["trend"].values[_o], "k-", lw=2.5, label="fitted curve")
        _a.set(yscale="log", ylabel="variance (donors 1–3)", title=f"{_name}: mean explains R² = {_f['r2']:.0%}")
    _ax[0].set(xlabel="mean log2 intensity")
    _m = fits["RNA-seq"]["mean"]
    _phi = np.median(((fits["RNA-seq"]["trend"] - _m) / _m**2)[_m > 100])
    _g = np.logspace(np.log10(_m.min()), np.log10(_m.max()), 100)
    _ax[1].plot(_g, _g + _phi * _g**2, "--", color="#4C72B0", lw=2, label=f"NB: μ + φμ², φ = {_phi:.3f}")
    _ax[1].set(xscale="log", xlabel="mean count")
    for _a in _ax:
        _a.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **RNA-seq:** the mean explains most of the variation in variance. Knowing a gene's
      expression level tells you most of what you need to know about its variance, and
      the NB curve $\mu + \phi\mu^2$ with a single $\phi$ already tracks the lowess curve.
    * **Array:** the curve is nearly flat; the mean explains very little. Genes differ in
      variance for reasons unrelated to their expression level.

    ## 3. Gene-specific variance estimates

    Three ways to estimate a gene's variance from donors 1–3:

    1. **Own:** the gene's sample variance $s_g^2$ (what a t-test uses).
    2. **Curve:** the fitted curve at the gene's mean, $t(\mu_g)$. Ignores the gene's own
       data.
    3. **Shrunk:** a weighted average of the two, weighting each by how much
       information it carries:

    $$\tilde s_g^2 = \frac{d_0\, t(\mu_g) + d\, s_g^2}{d_0 + d}$$

    Here $d = n - 1 = 2$ is the gene's own degrees of freedom, and $d_0$ ("prior degrees
    of freedom") measures how tightly genes cluster around the curve. If genes scatter
    around the curve no more than sampling noise predicts, $d_0 \to \infty$ and we use the
    curve. If they scatter a lot, $d_0$ is small and the gene's own estimate counts more.
    $d_0$ is estimated from the data (limma's method of moments). This is limma's
    empirical Bayes (`eBayes`, with a trend). DESeq2 does the same with the dispersion
    $\phi$ instead of the variance.
    """)
    return


@app.cell
def _(DF, fits, np, optimize, special):
    def prior_df(f):
        """limma-style moment estimate of d0 from scatter of log variances around the trend."""
        resid = np.log(f["var"]) - np.log(f["trend"])
        excess = np.var(resid) - special.polygamma(1, DF / 2)
        if excess <= 0:
            return np.inf
        return 2 * optimize.brentq(lambda h: special.polygamma(1, h) - excess, 1e-8, 1e8)


    for _f in fits.values():
        _f["d0"] = prior_df(_f)
        _f["shrunk"] = (_f["d0"] * _f["trend"] + DF * _f["var"]) / (_f["d0"] + DF)
    {_k: round(_f["d0"], 2) for _k, _f in fits.items()}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Estimated $d_0$ is printed above. The plot shows 40 random RNA-seq genes: each arrow goes from the
    gene's own variance to its shrunk variance.
    """)
    return


@app.cell
def _(SEQ_COLOR, fits, np, plt):
    _f = fits["RNA-seq"]
    _rng = np.random.default_rng(3)
    _pick = _rng.choice(len(_f["mean"]), 40, replace=False)
    _o = np.argsort(_f["mean"].values)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.scatter(_f["mean"], _f["var"], s=2, alpha=0.06, color=SEQ_COLOR)
    _ax.plot(_f["mean"].values[_o], _f["trend"].values[_o], "k-", lw=2, label="fitted curve")
    for _i in _pick:
        _x, _y0, _y1 = _f["mean"].iloc[_i], _f["var"].iloc[_i], _f["shrunk"].iloc[_i]
        _ax.annotate("", xy=(_x, _y1), xytext=(_x, _y0), arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1))
        _ax.plot(_x, _y0, "o", ms=3, color=SEQ_COLOR)
    _ax.set(xscale="log", yscale="log", xlabel="mean count", ylabel="variance",
            title=f"RNA-seq: own → shrunk (d₀ = {_f['d0']:.1f}, d = 2)")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which estimate is closest to the held-out donors?

    For each gene, compare each estimate with the variance from the held-out donors 4–6.
    The error is the median of $|\log_2(\text{estimate} / \text{held-out})|$ across genes:
    1 means a typical gene is off by 2-fold. The held-out variance is itself based on only 3
    donors, so no method gets close to 0; what matters is the comparison between rows.
    """)
    return


@app.cell
def _(fits, mo, np, pd):
    _rows = []
    for _name, _f in fits.items():
        _err = lambda est: np.median(np.abs(np.log2(est / _f["held_out"])))
        _rows.append({
            "technology": _name,
            "mean explains (R²)": f"{_f['r2']:.0%}",
            "prior df d₀": round(_f["d0"], 1),
            "error: own variance": round(_err(_f["var"]), 2),
            "error: curve": round(_err(_f["trend"]), 2),
            "error: shrunk": round(_err(_f["shrunk"]), 2),
        })
    mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Take-home**

    * A gene's own variance from a few replicates is the worst estimate in both
      technologies.
    * **RNA-seq:** the curve alone does about as well as shrinking, because the expression
      level predicts the variance so well. This is why DESeq2 and edgeR model
      $\operatorname{Var} = \mu + \phi\mu^2$, fit a dispersion trend across genes, and
      shrink each gene's $\phi$ toward it. voom uses the same kind of curve to give each
      observation a precision weight.
    * **Array:** the curve is nearly flat, so it adds little; shrinking toward it still
      helps by borrowing strength across genes. That is limma's moderated t-test.
    * Either way, the variance used in the test comes from **all genes**, not just the one
      being tested. With 3 replicates this matters more than the exact distribution.
    """)
    return


if __name__ == "__main__":
    app.run()
