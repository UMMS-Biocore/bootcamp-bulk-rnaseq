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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Count data and GLMs

    Where the negative binomial comes from (simulated qPCR, array and RNA-seq for the same
    biology), and what changes when we move from a linear model to a count GLM.

    The first cell loads the libraries.
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
    return COLORS, mo, np, pd, plt, sm, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same biology, three measurements

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

    **Controls:** Two sliders: the mean expression level of the gene and its biological variability (dispersion φ). Every plot below recomputes when you move them.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `sim_abundance` draws the true per-sample abundance λ from a Gamma distribution. The three `measure_*` functions turn λ into what each technology reports: a Ct value, a log2 array intensity, or a Poisson read count.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Draw 3000 biological replicates of the same gene, measure each with all three technologies, and plot the histograms, each on the scale the technology reports (Ct, log2 intensity, read counts). The black curve is the normal distribution with the same mean and SD, i.e. what a linear model assumes. The summary line compares the observed count variance with the Poisson and negative-binomial predictions.
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
    stats,
):
    _rng = np.random.default_rng(42)
    _mu, _phi = mu_slider.value, phi_slider.value
    _lam = sim_abundance(_rng, _mu, _phi, 3000)
    _ct = measure_qpcr(_rng, _lam)
    _arr = measure_array(_rng, _lam)
    _cnt = measure_seq(_rng, _lam)

    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.4))
    _bins = np.arange(-0.5, _cnt.max() + 1.5, max(1, int(np.ceil((_cnt.max() + 1) / 60))))
    for _a, _v, _b, _c, _t, _xl in (
        (_ax[0], _ct, 40, COLORS["qpcr"], "qPCR", "Ct"),
        (_ax[1], _arr, 40, COLORS["array"], "Microarray", "log2 intensity"),
        (_ax[2], _cnt, _bins, COLORS["seq"], "RNA-seq", "read count"),
    ):
        _a.hist(_v, bins=_b, density=True, color=_c, alpha=0.8)
        _xx = np.linspace(_v.min(), _v.max(), 300)
        _a.plot(_xx, stats.norm.pdf(_xx, _v.mean(), _v.std()), "k-", lw=1.5, label="normal, same mean & SD")
        _a.set(title=_t, xlabel=_xl, yticks=[])
    _ax[0].legend(frameon=False, fontsize=8)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Things to try**

    * Set μ to 0.3–3. The counts are a handful of integers piled up at zero, nothing like
      the normal curve.
    * Set μ to 30–300 with φ ≈ 0.2+. Counts are right-skewed: biological variation is
      multiplicative, so counts are roughly *log*-normal, not normal. The qPCR and array
      measurements are already on a log scale, so the same biology looks normal there.
    * Increase φ. Every technology gets wider (that's biology), but for counts the
      variance/mean ratio grows with μ: extra-Poisson variance is $\phi\mu^2$.
    * Low μ on the array: the signal sinks into background. High μ: saturation.
      Arrays have their own problems, they're just *continuous* problems.

    ### Mean–variance relationship across genes

    One gene is anecdotal. Now simulate thousands of genes with means spanning five orders
    of magnitude, each measured in *n* replicates, and plot how spread depends on level.

    **Controls:** Number of replicates per gene and a common dispersion φ for all simulated genes.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simulate 3000 genes with means from 0.1 to 10,000 reads, measure each in *n* replicates, then plot per-gene spread against per-gene mean, each on the scale the technology reports: count variance vs mean count (log–log axes, with Poisson and NB curves) and SD vs mean of the array's log2 intensity.
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

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].scatter(_m[_keep], _v[_keep], s=3, alpha=0.3, color=COLORS["seq"])
    _ax[0].plot(_grid, _grid, "k--", lw=1, label="Poisson: var = μ")
    _ax[0].plot(_grid, _grid + _phi * _grid**2, "k-", lw=1.5, label="NB: var = μ + φμ²")
    _ax[0].set(xscale="log", yscale="log", xlabel="mean count", ylabel="variance",
               title="RNA-seq counts")
    _ax[0].legend(frameon=False, fontsize=8)

    _ax[1].scatter(_arr.mean(1), _arr.std(1, ddof=1), s=3, alpha=0.3, color=COLORS["array"])
    _ax[1].set(xlabel="mean log2 intensity", ylabel="SD", title="Microarray, log2 intensity", ylim=(0, None))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
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

    ## From linear model to GLM for counts (preview)

    Keep the *same* linear predictor, change two things:

    $$y_{ij} \sim \text{NB}(\mu_{ij}, \phi_i), \qquad \log \mu_{ij} = \beta_0 + \beta_1 x_j \;(+\log s_j)$$

    * **Distribution:** negative binomial instead of normal, so $\operatorname{Var} = \mu + \phi\mu^2$
      (what we saw in Part 1).
    * **Link:** model $\log\mu$, so $\beta_1$ is a log fold change and fitted means stay positive.
    * $\log s_j$ is an *offset* for library size / size factor. No need to pre-normalize counts.

    This is the model inside **DESeq2** and **edgeR**.

    **Controls:** Baseline mean count, true log2FC, dispersion and group size for the count simulations below.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simulate NB counts for two groups. Fit (1) OLS on log2(count+1), which is the t-test, and (2) a negative-binomial GLM with log link and the true φ (`sm.GLM(..., family=NegativeBinomial)`). Compare the estimated log2FCs and p-values.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Many genes: false positives and power

    Simulate 4000 genes at the chosen baseline mean. Under the **null** (log2FC = 0) a
    calibrated test should give ~5% of p-values below 0.05. Under the **alternative**,
    the fraction below 0.05 is the power.

    For a two-group design the NB GLM has a closed form: $\hat\mu_g = \bar y_g$ and
    $\operatorname{SE}(\hat\beta_1)^2 = \sum_g \big[n_g \hat\mu_g / (1+\phi\hat\mu_g)\big]^{-1}$,
    so we can do this fast without calling a fitting routine 4000 times.

    Simulate 4000 null genes and 4000 truly changed genes. Test each with a t-test on log2(count+1) and with the closed-form NB Wald test, then report the fraction with p < 0.05.
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


@app.cell(hide_code=True)
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
