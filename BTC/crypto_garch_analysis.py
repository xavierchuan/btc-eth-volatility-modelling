import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model

# -----------------------------
# User settings
# -----------------------------
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"
TICKERS    = ["BTC-USD", "ETH-USD"]  # run both
DIST       = "normal"                 # options: 'normal', 't'
SHOW_PLOTS = False                    # do not block execution with figures

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = os.path.join(OUT_DIR, "figs")
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"Downloaded dataframe for {ticker} is empty. Check ticker or date range.")
    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Prefer 'Adj Close', fallback to 'Close'
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise KeyError(f"Neither 'Adj Close' nor 'Close' found in columns: {df.columns.tolist()}")
    df = df[[price_col]].rename(columns={price_col: "Price"}).copy()
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
    df.dropna(inplace=True)
    return df

def descriptive_and_jb(logret: pd.Series) -> pd.Series:
    desc = logret.describe()
    desc["skewness"] = logret.skew()
    desc["kurtosis"] = logret.kurtosis()
    jb_stat, jb_p, _, _ = jarque_bera(logret)
    desc["JB_stat"] = jb_stat
    desc["JB_pvalue"] = jb_p
    return desc

def fit_models(logret: pd.Series, dist: str = "normal") -> dict:
    results = {}
    # GARCH(1,1)
    am_garch = arch_model(logret, vol="GARCH", p=1, q=1, dist=dist)
    res_garch = am_garch.fit(disp="off")
    results["GARCH(1,1)"] = res_garch

    # EGARCH(1,1)
    am_egarch = arch_model(logret, vol="EGARCH", p=1, q=1, dist=dist)
    res_egarch = am_egarch.fit(disp="off")
    results["EGARCH(1,1)"] = res_egarch

    # GJR-GARCH(1,1)  -> use 'o=1' to add threshold term
    am_gjr = arch_model(logret, vol="GARCH", p=1, o=1, q=1, dist=dist)
    res_gjr = am_gjr.fit(disp="off")
    results["GJR-GARCH(1,1)"] = res_gjr

    return results

def model_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "LogLik": res.loglikelihood,
            "AIC": res.aic,
            "BIC": res.bic
        })
    return pd.DataFrame(rows).sort_values(["AIC","BIC"]).reset_index(drop=True)

def residual_diagnostics(res) -> pd.DataFrame:
    std_resid = res.std_resid
    # Ljung-Box on residuals and squared residuals
    lb1 = acorr_ljungbox(std_resid, lags=[12], return_df=True)
    lb2 = acorr_ljungbox(std_resid**2, lags=[12], return_df=True)
    out = pd.DataFrame({
        "LB(12)_resid_stat": lb1["lb_stat"].values,
        "LB(12)_resid_p": lb1["lb_pvalue"].values,
        "LB(12)_resid2_stat": lb2["lb_stat"].values,
        "LB(12)_resid2_p": lb2["lb_pvalue"].values
    })
    return out

def plot_acf_squared(logret: pd.Series, title: str, save_name: str | None = None):
    # 确保没有 NA
    x = (logret.dropna() ** 2).values
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_acf(x, lags=20, ax=ax)
    ax.set_title(title)
    fig.tight_layout()

    if SHOW_PLOTS:
        plt.show()
    if save_name:
        fig_path = os.path.join(FIG_DIR, save_name)
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {fig_path}")
    plt.close(fig)


def plot_conditional_vol(res, title: str, save_name: str | None = None):
    import matplotlib.pyplot as plt
    cond_vol = res.conditional_volatility

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(cond_vol.index, cond_vol.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Conditional Volatility")
    fig.tight_layout()

    if SHOW_PLOTS:
        plt.show()
    if save_name:
        fig_path = os.path.join(FIG_DIR, save_name)
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {fig_path}")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def run_for_ticker(ticker: str):
    print("="*80)
    print(f"Ticker: {ticker} | Period: {START_DATE} to {END_DATE}")
    print("="*80)

    df = download_prices(ticker, START_DATE, END_DATE)

    # Preliminary analysis
    desc = descriptive_and_jb(df["LogReturn"])
    print("\nDescriptive statistics & Jarque-Bera test:")
    print(desc.to_string())
    # Save descriptive stats & JB
    desc_df = desc.to_frame().T
    desc_path = os.path.join(OUT_DIR, f"{ticker}_descriptive_and_JB.csv")
    desc_df.to_csv(desc_path, index=False)
    print(f"\n[Saved] {desc_path}")

    plot_acf_squared(df["LogReturn"], f"ACF of Squared Returns - {ticker}", save_name=f"{ticker}_acf_squared.png")

    # Fit models
    results = fit_models(df["LogReturn"], dist=DIST)

    for name, res in results.items():
        summary_text = str(res.summary())
        safe_name = name.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
        log_path = os.path.join(LOG_DIR, f"{ticker}_{safe_name}_summary.txt")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(summary_text)
        print(f"[Saved] {log_path}")

    # Print summaries (optional, can be verbose)
    for name, res in results.items():
        print("\n" + "-"*60)
        print(f"{name} ({DIST}) summary for {ticker}:")
        print(res.summary())

    # Compare models by AIC/BIC
    cmp_table = model_comparison_table(results)
    print("\nModel comparison (lower is better):")
    print(cmp_table.to_string(index=False))
    # Save model comparison table
    cmp_path = os.path.join(OUT_DIR, f"{ticker}_model_comparison_AIC_BIC.csv")
    cmp_table.to_csv(cmp_path, index=False)
    print(f"[Saved] {cmp_path}")

    # Residual diagnostics for the best (by AIC)
    best_name = cmp_table.iloc[0]["Model"]
    best_res = results[best_name]
    diag = residual_diagnostics(best_res)
    print(f"\nResidual diagnostics for best model ({best_name}):")
    print(diag.to_string(index=False))
    # Save residual diagnostics for best model
    diag_path = os.path.join(OUT_DIR, f"{ticker}_residual_diagnostics_best_model.csv")
    diag.to_csv(diag_path, index=False)
    print(f"[Saved] {diag_path}")

    # Plot conditional volatility of the best model
    plot_conditional_vol(best_res, f"Conditional Volatility - {ticker} - {best_name}", save_name=f"{ticker}_cond_vol_{best_name.replace(' ', '_')}.png")

    print(f"\n[Done] {ticker} finished.\n")

def main():
    for t in TICKERS:
        try:
            run_for_ticker(t)
        except Exception as e:
            print(f"[Error] {t}: {e}")

if __name__ == "__main__":
    main()
