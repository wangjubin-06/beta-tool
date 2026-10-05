# beta_tool: Beta, Reference Assets, and Beta-Neutral Hedging

A detailed explanation of what `beta_tool` computes, what changes when the reference asset is not SPY, and what the hedging layer (`PortfolioHedge`) does mathematically.

> **Note on numbers.** All numeric examples below are *hypothetical* and chosen to make the arithmetic clean. They are not real market data or real `beta_tool` output.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $r_{a,t}$ | Simple return of the **target** asset (or portfolio) on day $t$ |
| $r_{m,t}$ | Simple return of the **reference** asset on day $t$ (SPY, QQQ, a sector ETF, TLT, ...) |
| $\alpha, \beta$ | Intercept and slope of the regression |
| $\varepsilon_t$ | Regression residual |
| $\sigma_a, \sigma_m$ | Standard deviations of $r_a$ and $r_m$ |
| $\rho$ | Correlation between $r_a$ and $r_m$ |
| $R^2$ | Fraction of $r_a$'s variance explained by the regression ($=\rho^2$ with one regressor) |
| $V$ | Dollar value of the position being hedged |

---

## 2. What beta actually is

The `Beta` tool runs an ordinary least squares regression of the target's returns on a reference asset's returns:

$$
r_{a,t} = \alpha + \beta\, r_{m,t} + \varepsilon_t
$$

The OLS slope is

$$
\hat\beta = \frac{\operatorname{Cov}(r_a, r_m)}{\operatorname{Var}(r_m)} = \rho \,\frac{\sigma_a}{\sigma_m}
$$

**Interpretation.** A beta of 1.35 means: on average, when the reference moves by 1%, the target moves by about 1.35% in the same direction (plus noise).

Nothing in this maths is specific to SPY. "Market beta" is just the special case where the reference is a market-index proxy. **Beta is always relative to the reference series you choose.**

### Variance decomposition

Taking the variance of the regression equation (and using the fact that $\varepsilon$ is uncorrelated with $r_m$ by construction):

$$
\underbrace{\operatorname{Var}(r_a)}_{\text{total risk}} = \underbrace{\beta^2 \operatorname{Var}(r_m)}_{\text{systematic (explained by reference)}} + \underbrace{\operatorname{Var}(\varepsilon)}_{\text{residual}}
$$

and therefore

$$
R^2 = \frac{\beta^2 \sigma_m^2}{\sigma_a^2} = \rho^2, \qquad \operatorname{Var}(\varepsilon) = (1-R^2)\,\sigma_a^2
$$

This decomposition is the key to understanding both "which reference should I use?" and "how good can the hedge be?".

---

## 3. Using a reference other than SPY

Changing the reference changes the *question* being asked.

| Reference | Question beta answers | Typical use |
|---|---|---|
| SPY (broad market) | How much does the asset move with the overall US equity market? | Market-risk hedge, CAPM-style exposure |
| QQQ | Exposure to the growth / large-cap tech tilt | Tech-heavy stocks and portfolios |
| Sector ETF (XLF, XLE, XLK, ...) | Exposure to the asset's own industry | Stripping out industry risk; usually higher $R^2$ for single stocks |
| Bond ETF (TLT, IEF, ...) | Rate sensitivity, a market-implied duration-like number | Fixed-income-style exposure, rate-risk hedging |
| Gold, USD, oil, BTC | Exposure to a specific macro driver | Macro sensitivity analysis |
| A peer / competitor stock | Co-movement with a specific name | Pairs-style relative hedges |

Three practical points:

1. **Different references give different betas for the same asset.** That is not a contradiction; they measure different exposures.
2. **Beta and fit are separate things.** Beta is the *slope*; $R^2$ tells you how well the reference explains the asset. A stock can have $\beta \approx 1$ to two references, but one fit may be far tighter than the other.
3. **Pick the reference by what you want to neutralise.** If you want to remove market risk, use a market proxy. If you want to remove as much total variance as possible with one instrument, pick the reference with the highest $R^2$ (see Section 5).

### Multiple references at once (`MultiBeta`, `FactorsRegression`)

With $K$ references, the regression becomes

$$
r_{a,t} = \alpha + \sum_{k=1}^{K}\beta_k\, r_{k,t} + \varepsilon_t, \qquad \hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top y
$$

Each $\beta_k$ is now a **partial** effect: the sensitivity to reference $k$ *holding the others fixed*. When references are correlated, partial betas differ from single-reference betas.

*Hypothetical illustration:* a tech stock has a univariate $\beta_{SPY}=1.20$ and $\beta_{QQQ}=1.00$. Regressed on both together, SPY and QQQ overlap so heavily that the result might look like $\beta_{SPY}=0.40$, $\beta_{QQQ}=0.80$. The stock did not change; the *attribution* between two overlapping references did. This is multicollinearity, and it is why multi-factor regressions should use reasonably distinct factors (as in Fama-French: market, size, value, momentum, ...).

`FactorsRegression` is the same idea applied to ETF proxies or Fama-French factors, and `PortfolioBeta` applies it to a weighted portfolio's return series.

### What HAC (Newey-West) does

Daily financial returns are often heteroskedastic and mildly autocorrelated, so classic OLS standard errors can be too small. HAC-robust estimation **leaves $\hat\beta$ unchanged** and only changes the standard errors, p-values and confidence intervals. It is a sensible default for judging how precisely beta is estimated.

---

## 4. What the hedging layer does

`PortfolioHedge` builds a **beta-neutral hedge** with respect to the hedge instrument you choose. It is not automatically "market-neutral": it is neutral to whatever you hedge *with*.

### 4.1 The hedged return series

$$
h_t = r_{a,t} - \beta\, r_{H,t}
$$

where $r_{H,t}$ is the return of the hedge instrument and $\beta$ is the target's beta to that instrument. Simple returns are used (not log returns) because simple returns aggregate linearly across assets at a point in time, which is what a dollar-weighted long/short position needs.

**In dollars:** hold a position worth $V$ and short a notional of

$$
N_H = \beta \cdot V
$$

of the hedge instrument. The combined daily P&L is $V\,r_a - N_H\, r_H = V\,(r_a - \beta r_H) = V\,h$.

### 4.2 Why this is "beta-neutral"

The covariance of the hedged series with the hedge instrument is

$$
\operatorname{Cov}(h, r_H) = \operatorname{Cov}(r_a, r_H) - \beta \operatorname{Var}(r_H) = 0
$$

when $\beta = \operatorname{Cov}(r_a, r_H)/\operatorname{Var}(r_H)$. So the hedged position has zero regression-beta to the hedge instrument.

### 4.3 It is also the minimum-variance hedge

Let the hedge ratio be a free number $b$. Then

$$
\operatorname{Var}(r_a - b\,r_H) = \sigma_a^2 - 2b\operatorname{Cov}(r_a,r_H) + b^2\sigma_H^2
$$

Setting the derivative with respect to $b$ to zero:

$$
\frac{d}{db}\operatorname{Var}(r_a - b\,r_H) = -2\operatorname{Cov}(r_a, r_H) + 2b\,\sigma_H^2 = 0
$$

which gives the optimal hedge ratio

$$
b_{\text{opt}} = \frac{\operatorname{Cov}(r_a, r_H)}{\sigma_H^2} = \beta
$$

So for a single hedge instrument, **beta-neutral and minimum-variance coincide**, and the remaining variance is

$$
\operatorname{Var}(h) = (1 - R^2)\,\sigma_a^2
$$

i.e. the hedge removes a fraction $R^2$ of the target's variance and the rest is residual risk.

### 4.4 Static vs rolling hedge ratios

| | Static | Rolling |
|---|---|---|
| Beta estimated from | Data **up to the backtest start date only** | A trailing window ($W$ days) |
| Hedge ratio over time | One constant $\beta$ | Piecewise-constant $\beta_k$, updated at each rebalance |
| Strength | Simple, low turnover | Adapts if the relationship drifts |
| Risk | Goes stale if beta changes | More estimation noise and more trading |

Two implementation details matter for honesty of the backtest:

- **No look-ahead.** A static $\beta$ estimated on the full sample would use information from the future. Estimating it only on data before `backtest_start_date` avoids this. For rolling hedges, the beta used on day $t$ is the *lagged* rolling estimate (`shift(1)`), then held (forward-filled) until the next rebalance date.
- **Warm-up.** The rolling window needs $W$ days of history before the first hedged day, so the tool pulls $W$ extra trading days before the analysis period. This avoids NaNs at the start without sneaking in a full-sample beta.

A default rebalance frequency of $\max(\lfloor W/6 \rfloor, \text{floor})$ keeps the conventional 6:1 window-to-rebalance ratio. For example, $W=126$ trading days rebalances roughly every 21 trading days.

For a portfolio, the target return is derived from the portfolio value series, $r_{p,t} = V_t/V_{t-1} - 1$, so the hedged and unhedged series are built on the same units-based accounting.

### 4.5 Hedging with several instruments

With $K$ hedge instruments, the minimum-variance hedge vector is the multiple-regression coefficient vector:

$$
\mathbf{b}_{\text{opt}} = \Sigma_{HH}^{-1}\,\Sigma_{Ha} = \hat{\boldsymbol{\beta}}_{\text{MultiBeta}}
$$

and the hedged series is

$$
h_t = r_{a,t} - \mathbf{b}_{\text{opt}}^{\top}\,\mathbf{r}_{H,t}
$$

Here $\Sigma_{HH}$ is the covariance matrix of the hedge instruments and $\Sigma_{Ha}$ is the vector of their covariances with the target. This is how you would neutralise, say, market **and** rate exposure together. (The single-instrument formula in 4.1 is the case $K=1$.)

---

## 5. Worked examples (hypothetical numbers)

Setup: a single stock with daily volatility $\sigma_a = 1.8\%$. Position size $V$ = &#36;100,000.

### Example A: Hedging with SPY

Assume $\sigma_{SPY} = 1.0\%$ and $\rho = 0.75$.

$$
\beta_{SPY} = 0.75 \times \frac{1.8}{1.0} = 1.35, \qquad R^2 = 0.75^2 = 0.5625
$$

**Hedge size:** short &#36;1.35 $\times$ &#36;100,000 = &#36;135,000 of SPY.

**One-day check.** Suppose the stock returns $+2.0\%$ and SPY returns $+1.0\%$:

- Stock P&L: $+$ &#36;$2{,}000$
- SPY short P&L: $-$ &#36;$135{,}000 \times 1.0\% =$ $-$ &#36;$1{,}350$
- Net: $+$ &#36;$650$, i.e. $h = 2.0\% - 1.35\times 1.0\% = 0.65\%$ of $V$

**Risk after hedging:**

$$
\sigma_h = \sigma_a\sqrt{1-R^2} = 1.8\%\times\sqrt{0.4375} \approx 1.19\% \text{ per day}
$$

Annualised ($\times\sqrt{252}$): about $28.6\% \to 18.9\%$. The hedge removed $56\%$ of the variance.

### Example B: Hedging the same stock with its sector ETF

Assume $\sigma_{sector} = 1.6\%$ and $\rho = 0.90$.

$$
\beta_{sector} = 0.90\times\frac{1.8}{1.6} = 1.0125, \qquad R^2 = 0.81
$$

**Hedge size:** short about &#36;$101{,}250$ of the sector ETF.

$$
\sigma_h = 1.8\%\times\sqrt{0.19} \approx 0.78\% \text{ per day} \;\;(\approx 12.5\% \text{ annualised})
$$

### Comparison

| Hedge instrument | $\beta$ | $R^2$ | Residual daily vol | Annualised vol | Variance removed |
|---|---|---|---|---|---|
| None | n/a | n/a | 1.80% | 28.6% | 0% |
| SPY | 1.35 | 0.5625 | 1.19% | 18.9% | 56% |
| Sector ETF | 1.0125 | 0.81 | 0.78% | 12.5% | 81% |

**Takeaway.** Both hedges are "beta-neutral" to their own instrument. The sector hedge is more effective *for total risk* because it explains more of the stock's variance. The SPY hedge, however, leaves you with pure market-neutral exposure, whereas the sector hedge leaves residual *market* exposure of the stock that the sector does not capture. Which is better depends on what risk you are trying to remove.

### Example C: Rolling vs static

Say $W = 126$ trading days, rebalanced every 21 days. On rebalance date $k$ the tool uses the rolling beta that was available the *day before*, so $\beta_k = 1.35$ might become $\beta_{k+1} = 1.50$ after a volatile stretch, and the short SPY notional is adjusted from &#36;$135{,}000$ to &#36;$150{,}000$ at that date and then held constant until the next rebalance. A static hedge would stay at $1.35$ throughout, which is better if the true beta is stable and worse if it drifts.

### Example D: Rate beta (fixed-income flavour)

Regress a corporate bond fund's returns on an intermediate Treasury ETF (e.g. IEF). A slope of, say, 0.9 says the fund moves about 0.9% for each 1% move in the Treasury ETF, a market-implied rate sensitivity. Hedging with $N_H = 0.9\,V$ of the Treasury ETF removes most of the *rate* exposure and leaves credit-spread risk and idiosyncratic risk as the residual.

---

## 6. Limitations and caveats

1. **Neutral only to the chosen instrument.** Everything else (sector, other factors, idiosyncratic risk) remains.
2. **Residual risk is irreducible by this hedge.** In-sample it is $(1-R^2)\sigma_a^2$; out-of-sample it is usually worse.
3. **Beta is estimated, not known.** Its standard error is non-trivial on short windows. HAC standard errors show this; they do not shrink it.
4. **Beta is unstable.** Regimes (e.g. 2008, March 2020, rate shocks) change relationships. Rolling windows help but add noise: short windows are noisy, long windows are slow to adapt.
5. **Intercept ($\alpha$) is not hedged.** The hedge removes common-factor returns; it does not add return and can subtract it.
6. **Frictions are outside the simple model.** Real hedges face transaction costs, borrow costs, tracking error between the proxy and the true exposure, and hedge-ratio rounding.
7. **Correlation is not beta.** Two assets can be highly correlated with very different betas (depending on relative volatility). Hedge sizing needs $\beta$, while $R^2$ tells you how good the hedge will be.
8. **Multiple correlated hedge instruments can give unstable weights.** If you hedge with several overlapping ETFs, the individual hedge ratios can swing a lot even when the combined hedge is fine.

---

## 7. How the pieces fit together in `beta_tool`

| Component | Role |
|---|---|
| `Beta` | Two-asset OLS beta, optional HAC, optional fixed-window rolling beta |
| `MultiBeta` | N-asset regression (partial betas) |
| `PortfolioBeta` | Weighted portfolio returns regressed on asset(s) or a benchmark |
| `FactorsRegression` | Asset returns vs ETF proxies or Fama-French factors |
| `PortfolioHedge` | Builds hedged vs unhedged series using static or rolling beta, and backtests the result |

Data comes from Tiingo (requires `TIINGO_API_KEY`) with full history cached as parquet files.
