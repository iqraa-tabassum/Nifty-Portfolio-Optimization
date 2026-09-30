"""
Advanced Portfolio Optimization of Nifty Stocks
------------------------------------------------
Monte Carlo + exact optimizer (scipy) + Ledoit-Wolf covariance + weight cap
+ walk-forward backtest with quarterly rebalancing and transaction costs
+ risk metrics (Sharpe, Max Drawdown, VaR, CVaR)
+ comparison vs Equal-Weight and Nifty 50.

Install once:  pip install yfinance pandas numpy matplotlib scipy scikit-learn
Run:           python advanced_portfolio_optimizer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ---------------- SETTINGS ----------------
TICKERS = {                       # 10 stocks, different sectors
    "INFY.NS": "IT", "TCS.NS": "IT",
    "HDFCBANK.NS": "Bank", "RELIANCE.NS": "Energy",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG",
    "SUNPHARMA.NS": "Pharma", "MARUTI.NS": "Auto",
    "LT.NS": "Infra", "BHARTIARTL.NS": "Telecom",
}
BENCHMARK = "^NSEI"              # Nifty 50 index
YEARS = 7
RF = 0.07                        # risk-free rate (~Indian 10-yr yield)
CAP = 0.30                       # max weight in any one stock
LOOKBACK = 504                   # ~2 years of data used to decide weights
REBAL = 63                       # rebalance every ~quarter
TC = 0.001                       # 0.1% transaction cost on traded amount
N_SIM = 10000                    # Monte Carlo portfolios
DAYS = 252
USE_SYNTHETIC = os.environ.get("SYNTH") == "1"   # only for offline testing
if USE_SYNTHETIC:
    matplotlib.use("Agg")

np.random.seed(42)


# ---------------- DATA ----------------
def load_data():
    names = list(TICKERS)
    if USE_SYNTHETIC:
        n, k = 1750, len(names)
        base = np.random.randn(n, 1) * 0.008
        r = 0.6 * base + np.random.randn(n, k) * 0.010 + 0.0005
        stocks = pd.DataFrame(r, columns=names,
                              index=pd.bdate_range("2019-09-01", periods=n))
        bench = pd.Series(r.mean(axis=1) * 0.9 + np.random.randn(n) * 0.002,
                          index=stocks.index)
        return stocks, bench
    import yfinance as yf
    px = yf.download(names + [BENCHMARK], period=f"{YEARS}y",
                     auto_adjust=True)["Close"].dropna()
    rets = px.pct_change().dropna()
    return rets[names], rets[BENCHMARK]


# ---------------- OPTIMIZERS ----------------
def shrunk_cov(train):
    """Ledoit-Wolf covariance: more stable than the plain sample covariance."""
    return LedoitWolf().fit(train.values).covariance_ * DAYS


def max_sharpe_weights(train):
    mu = train.mean().values * DAYS
    cov = shrunk_cov(train)
    n = len(mu)
    neg_sharpe = lambda w: -(w @ mu - RF) / np.sqrt(w @ cov @ w)
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, CAP)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x if res.success else np.ones(n) / n


def equal_weights(train):
    return np.ones(train.shape[1]) / train.shape[1]


# ---------------- BACKTEST ----------------
def backtest(rets, strategy):
    """Walk-forward: at each rebalance date, use ONLY past data to pick weights."""
    n = rets.shape[1]
    w = np.zeros(n)
    out, w_log = [], {}
    for i in range(LOOKBACK, len(rets)):
        cost = 0.0
        if (i - LOOKBACK) % REBAL == 0:
            w_new = strategy(rets.iloc[i - LOOKBACK:i])
            cost = np.abs(w_new - w).sum() * TC
            w = w_new
            w_log[rets.index[i]] = w_new
        out.append(rets.iloc[i].values @ w - cost)
    idx = rets.index[LOOKBACK:]
    return pd.Series(out, index=idx), pd.DataFrame(w_log, index=rets.columns).T


# ---------------- METRICS ----------------
def metrics(r):
    growth = (1 + r).cumprod()
    years = len(r) / DAYS
    cagr = growth.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(DAYS)
    sharpe = (cagr - RF) / vol
    mdd = (growth / growth.cummax() - 1).min()
    var95 = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean()
    return {"CAGR %": cagr * 100, "Volatility %": vol * 100, "Sharpe": sharpe,
            "Max Drawdown %": mdd * 100, "VaR 95% (1-day) %": var95 * 100,
            "CVaR 95% (1-day) %": cvar95 * 100}


# ---------------- MAIN ----------------
stocks, bench = load_data()
print(f"Data: {stocks.shape[0]} days, {stocks.shape[1]} stocks")

# 1) Monte Carlo vs exact optimizer on the latest lookback window
window = stocks.iloc[-LOOKBACK:]
mu = window.mean().values * DAYS
cov = shrunk_cov(window)
sims = np.random.dirichlet(np.ones(len(mu)), N_SIM)
sims = sims[sims.max(axis=1) <= CAP]                    # respect the cap
sim_ret = sims @ mu
sim_risk = np.sqrt(np.einsum("ij,jk,ik->i", sims, cov, sims))
sim_sharpe = (sim_ret - RF) / sim_risk
best = sim_sharpe.argmax()

w_exact = max_sharpe_weights(window)
ex_ret, ex_risk = w_exact @ mu, np.sqrt(w_exact @ cov @ w_exact)
print(f"\nMonte Carlo best Sharpe : {sim_sharpe[best]:.3f} "
      f"({len(sims)} valid portfolios)")
print(f"Exact optimizer Sharpe  : {(ex_ret - RF) / ex_risk:.3f}")
print("\nExact max-Sharpe weights (capped at "
      f"{CAP:.0%}):")
for t, wt in sorted(zip(TICKERS, w_exact), key=lambda x: -x[1]):
    print(f"  {t:<14}{TICKERS[t]:<8}{wt * 100:5.1f}%")

# 2) Walk-forward backtest
opt_ret, opt_w = backtest(stocks, max_sharpe_weights)
eq_ret, _ = backtest(stocks, equal_weights)
nifty_ret = bench.loc[opt_ret.index]

table = pd.DataFrame({"Optimized (Max Sharpe)": metrics(opt_ret),
                      "Equal Weight": metrics(eq_ret),
                      "Nifty 50": metrics(nifty_ret)}).round(2)
print("\n=== OUT-OF-SAMPLE BACKTEST RESULTS ===")
print(table.to_string())
table.to_csv("backtest_results.csv")

# ---------------- CHARTS ----------------
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

sc = ax[0, 0].scatter(sim_risk * 100, sim_ret * 100, c=sim_sharpe,
                      cmap="viridis", s=6)
ax[0, 0].scatter(ex_risk * 100, ex_ret * 100, c="red", marker="*", s=300,
                 label="Exact optimizer (scipy)")
ax[0, 0].scatter(sim_risk[best] * 100, sim_ret[best] * 100, c="orange",
                 marker="P", s=120, label="Best Monte Carlo")
ax[0, 0].set(title="Efficient Frontier (capped weights)",
             xlabel="Risk %", ylabel="Return %")
ax[0, 0].legend()
fig.colorbar(sc, ax=ax[0, 0], label="Sharpe")

for name, r in [("Optimized", opt_ret), ("Equal Weight", eq_ret),
                ("Nifty 50", nifty_ret)]:
    ax[0, 1].plot((1 + r).cumprod(), label=name)
ax[0, 1].set(title="Growth of Rs 1 (out-of-sample)")
ax[0, 1].legend()

for name, r in [("Optimized", opt_ret), ("Equal Weight", eq_ret),
                ("Nifty 50", nifty_ret)]:
    g = (1 + r).cumprod()
    ax[1, 0].plot((g / g.cummax() - 1) * 100, label=name)
ax[1, 0].set(title="Drawdown %")
ax[1, 0].legend()

ax[1, 1].stackplot(opt_w.index, opt_w.T.values * 100, labels=opt_w.columns)
ax[1, 1].set(title="Optimizer weights over time (%)")
ax[1, 1].legend(fontsize=6, loc="upper left", ncol=2)

plt.tight_layout()
plt.savefig("advanced_results.png", dpi=150)
if not USE_SYNTHETIC:
    plt.show()h
