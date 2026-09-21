# Methodology

## Data and Returns

The project uses daily adjusted-close returns for the fixed 22-stock universe. A date is retained only when adjusted prices are available for every asset. For asset \(i\) on day \(t\),

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1.
$$

## Covariance Estimation

For a 504-observation return window, sample covariance is

$$
\hat\Sigma_{\mathrm{sample}} = \frac{1}{T-1}\sum_{t=1}^{T}(r_t-\bar r)(r_t-\bar r)^\top.
$$

EWMA uses normalized exponentially decaying weights, with the newest return receiving the greatest weight:

$$
\tilde w_t = \lambda^{\mathrm{age}_t}, \qquad
w_t = \frac{\tilde w_t}{\sum_s \tilde w_s}, \qquad
\lambda = 0.94.
$$

The implementation estimates a weighted mean and demeaned covariance,

$$
\mu_w = \sum_t w_t r_t, \qquad
\hat\Sigma_{\mathrm{EWMA}} = \sum_t w_t(r_t-\mu_w)(r_t-\mu_w)^\top.
$$

Ledoit-Wolf shrinks the empirical covariance toward a scaled identity target:

$$
\hat\Sigma_{\mathrm{LW}} = (1-\delta)\hat\Sigma_{\mathrm{sample}} + \delta\mu I.
$$

The shrinkage intensity \(\delta\) is fitted by scikit-learn's `LedoitWolf` estimator.

## Portfolio Construction

For each covariance estimate, the optimizer minimizes predicted portfolio variance,

$$
\min_w \quad w^\top\hat\Sigma w,
$$

subject to full investment, long-only weights, a 15% stock cap, and a 25% sector cap:

$$
\sum_i w_i=1, \qquad 0\leq w_i\leq0.15, \qquad \sum_{i\in s}w_i\leq0.25.
$$

The equal-weight portfolio assigns \(1/22\) to each asset and serves as a benchmark.

## Backtest Timing and Weight Drift

At each month-end, the covariance estimate uses the previous 504 daily returns. The resulting target is held over the following complete month. Daily portfolio return is

$$
r_{p,t}=\sum_i w_{i,t}r_{i,t}.
$$

After each holding-day return, weights drift according to

$$
w_{i,t+1}=\frac{w_{i,t}(1+r_{i,t})}{1+r_{p,t}}.
$$

The drifted weights immediately before the next rebalance are used in the turnover calculation.

## Forecast Evaluation

The annualized predicted volatility at a rebalance is

$$
\hat\sigma_{p,t}=\sqrt{252\,w_t^\top\hat\Sigma_t w_t}.
$$

Realized volatility is the annualized sample standard deviation of daily returns in the following holding month:

$$
\sigma_{\mathrm{realized},t}=SD(r_{p,t})\sqrt{252}.
$$

Forecast error is \(e_t=\hat\sigma_t-\sigma_{\mathrm{realized},t}\). The reported metrics are

$$
MAE=\frac{1}{N}\sum_t|e_t|, \qquad
RMSE=\sqrt{\frac{1}{N}\sum_t e_t^2}, \qquad
Bias=\frac{1}{N}\sum_t e_t.
$$

The underprediction rate is the share of holding periods with \(e_t<0\). The primary high-volatility diagnostic uses the upper quartile of the median realized volatility across GMV portfolios; a separate robustness check uses the upper quartile of Equal Weight realized volatility.

## Turnover and Costs

One-sided turnover at rebalance \(t\) is

$$
Turnover_t=\frac{1}{2}\sum_i\left|w^{\mathrm{target}}_{i,t}-w^{\mathrm{pretrade}}_{i,t}\right|.
$$

For a cost assumption of \(c\) basis points per dollar traded, the one-time deduction is

$$
Cost_t=2\times Turnover_t\times\frac{c}{10{,}000}.
$$

The first allocation has no prior portfolio and therefore no turnover cost. The analysis reports 5, 10, and 20 bps assumptions.

## Tail-Risk Evaluation

Stage 7 extends the realized-risk analysis to one-day downside-tail diagnostics using the saved, gross Stage 4 out-of-sample portfolio returns. It does not alter the optimizer, covariance estimators, portfolio returns, or transaction-cost analysis.

For portfolio return \(r_{p,t}\), the loss convention is

$$
L_t=-r_{p,t}.
$$

On each forecast date \(t\), the calculation uses only the preceding 252 observed portfolio returns, \(t-252\) through \(t-1\). The next observed return at \(t\) is then used only for backtesting. For confidence level \(\alpha\), historical VaR is the empirical \(\alpha\)-quantile of the 252 historical losses:

$$
VaR_{\alpha,t}=Q^{\mathrm{higher}}_{\alpha}(L_{t-252},\ldots,L_{t-1}).
$$

The `higher` convention selects the next empirical order statistic rather than interpolating between losses. VaR is reported directly from the loss quantile and is not floored at zero; an unusual all-positive historical-return window can therefore produce a non-positive reported VaR.

Expected Shortfall is the average loss at or beyond that threshold:

$$
ES_{\alpha,t}=\operatorname{mean}(L_j\mid L_j\geq VaR_{\alpha,t}).
$$

A VaR exceedance occurs when the realized loss is strictly greater than the forecast VaR, \(L_t>VaR_{\alpha,t}\). The backtest compares the empirical exceedance rate with its expected rate, \(1-\alpha\), and reports the Kupiec unconditional-coverage likelihood-ratio statistic and its one-degree-of-freedom p-value.

Historical simulation is a tail-risk diagnostic, not evidence that GMV improves returns. It assumes the trailing return distribution remains relevant, can react slowly to regime changes, and does not assess clustered exceedances. With a 252-day window, the 99% tail is supported by only a few historical observations, so its VaR and ES estimates are especially discrete and unstable.

## Concentration

Portfolio concentration is measured by the Herfindahl-Hirschman index,

$$
HHI=\sum_i w_i^2,
$$

and effective holdings are \(N_{\mathrm{eff}}=1/HHI\). The latter is the number of equal-weight holdings with the same concentration.
