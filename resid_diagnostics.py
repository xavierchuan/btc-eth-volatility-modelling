import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

# -----------------------------
# Settings
# -----------------------------
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"
TICKERS    = ["BTC-USD", "ETH-USD"]
DIST       = "normal"

OUT_DIR = "results/figs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Helper: 下载并计算 log return
# -----------------------------
def download_logreturns(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df[[price_col]].rename(columns={price_col: "Price"}).copy()
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
    return df["LogReturn"].dropna()

# -----------------------------
# Helper: 拟合最优模型（BTC 用 GJR-GARCH，ETH 用 EGARCH）
# -----------------------------
def fit_best_model(ticker: str, logret: pd.Series):
    if ticker == "BTC-USD":
        am = arch_model(logret, vol="GARCH", p=1, o=1, q=1, dist=DIST)  # GJR-GARCH
    else:
        am = arch_model(logret, vol="EGARCH", p=1, q=1, dist=DIST)      # EGARCH
    res = am.fit(disp="off")
    return res

# -----------------------------
# Helper: 绘制残差 ACF
# -----------------------------
def plot_resid_acf(res, ticker: str):
    std_resid = res.std_resid.dropna()

    # 1. ACF of residuals
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_acf(std_resid, lags=20, ax=ax)
    ax.set_title(f"{ticker} - Residual ACF")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{ticker}_resid_acf.png"), dpi=200)
    plt.close(fig)

    # 2. ACF of squared residuals
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_acf(std_resid**2, lags=20, ax=ax)
    ax.set_title(f"{ticker} - Squared Residuals ACF")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{ticker}_resid2_acf.png"), dpi=200)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    for t in TICKERS:
        logret = download_logreturns(t)
        res = fit_best_model(t, logret)
        plot_resid_acf(res, t)
        print(f"[Saved] residual diagnostics for {t}")

if __name__ == "__main__":
    main()