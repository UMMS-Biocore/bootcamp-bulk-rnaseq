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
    # 2. The t-test is a linear model

    Comparing two group means with a t-test is the same as fitting
    $y = \beta_0 + \beta_1 x + \varepsilon$ and testing $\beta_1 = 0$. Thinking in models
    lets us add covariates such as batch. Everything here is simulated, so we know the truth.

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
    return COLORS, mo, np, pd, plt, sm, smf, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two-sample t-test *is* a linear model

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

    **Controls:** Sample size per group, the true difference between groups (β₁), the noise SD, and a random seed to draw a new data set.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Draw *n* log-expression values per group from normal distributions whose means differ by the chosen log2FC. Store them in a table with an explicit 0/1 column `x`, which becomes the design-matrix column.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run Student's t-test (`scipy.stats.ttest_ind`, `equal_var=True`) and fit OLS of y on [1, x] (`statsmodels`). Plot the data with the fitted line, whose slope is β₁, and show both results side by side with the design matrix.
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


@app.cell(hide_code=True)
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

    ### Why bother rewriting a t-test as a model? Because experiments have structure.

    Suppose half the samples were processed on Monday and half on Friday (batch).
    The t-test has nowhere to put that. The model just gets another column:

    $$y_i = \beta_0 + \beta_1\,\text{treated}_i + \beta_2\,\text{batch}_i + \varepsilon_i$$

    **Controls:** Batch effect size, whether batch is balanced across groups or partially confounded with treatment, group size, true log2FC, and seed.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simulate expression = baseline + treatment effect + batch effect + noise. Compare the t-test (ignores batch) with `ols('y ~ treated + C(batch)')` on the estimated log2FC and p-value.
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


if __name__ == "__main__":
    app.run()
