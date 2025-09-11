# Crypto Volatility Modelling with GARCH-type Models

This project analyzes and compares the volatility dynamics of Bitcoin (BTC) and Ethereum (ETH) using three classic GARCH-type models: **GARCH(1,1)**, **EGARCH(1,1)**, and **GJR-GARCH(1,1)**. It provides a modular pipeline for log-return transformation, model fitting, residual diagnostics, and comparative visualization.

## 🧠 Objective

To assess and visualize the performance of different GARCH-family models on the volatility clustering behavior of BTC and ETH, using historical daily closing prices.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `crypto_garch_analysis.py` | CLI entry point to fit GARCH models and run diagnostics |
| `log_return.py` | Utility to compute log-returns from price series |
| `resid_diagnostics.py` | Ljung–Box and Jarque–Bera tests for model residuals |
| `plot_model_comparison.py` | Visual comparison of fitted models with actual volatility |
| `requirements.txt` | Python dependencies (e.g., `arch`, `pandas`, `yfinance`, etc.) |

---

## 📈 Supported Models

- `GARCH(1,1)`
- `EGARCH(1,1)`
- `GJR-GARCH(1,1)`

All models are fitted using Maximum Likelihood Estimation (MLE) under normal distribution assumptions.

---

## 🚀 How to Use

Install dependencies:

```bash
pip install -r requirements.txt

Run full volatility analysis via CLI:
python crypto_garch_analysis.py --ticker BTC-USD --start 2020-01-01 --end 2024-01-01 --model all

Options:
	•	--ticker: Specify cryptocurrency (e.g., BTC-USD, ETH-USD)
	•	--model: Choose model (garch, egarch, gjr or all)
	•	--start, --end: Time window for historical data
	•	--output: Save plots and stats (optional)

⸻

🧪 Model Evaluation

Each model output includes:
	•	Estimated parameters (ω, α, β, etc.)
	•	Residual diagnostics: Ljung–Box Q-statistic, Jarque–Bera test
	•	Volatility forecasts and confidence intervals
	•	AIC/BIC comparison (optional)

⸻

📊 Sample Output
GARCH(1,1) Forecast (BTC)
Model Comparison (ETH)

📌 Highlights
	•	Supports modular GARCH-type volatility modelling
	•	End-to-end CLI and Python API
	•	Residual testing and volatility forecast plotting
	•	Easily extended to other assets or models (e.g., TARCH, APARCH)

⸻

🧠 Author

**Xiaochuan Li**
[LinkedIn](https://www.linkedin.com/in/xiaochuan-li-finance/) | [GitHub](https://github.com/xavierchuan)

⸻

📄 License

This project is licensed under the MIT License.
