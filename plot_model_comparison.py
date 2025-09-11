import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Load BTC & ETH model comparison CSV
btc = pd.read_csv(os.path.join(RESULTS_DIR, "BTC-USD_model_comparison_AIC_BIC.csv"))
eth = pd.read_csv(os.path.join(RESULTS_DIR, "ETH-USD_model_comparison_AIC_BIC.csv"))

# Add asset column
btc["Asset"] = "BTC"
eth["Asset"] = "ETH"

# Combine
df = pd.concat([btc, eth], ignore_index=True)

# -----------------------------
# Plot AIC comparison
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, asset in zip(axes, ["BTC", "ETH"]):
    subset = df[df["Asset"] == asset]
    ax.bar(subset["Model"], subset["AIC"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(f"{asset} - AIC Comparison")
    ax.set_ylabel("AIC")
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Model Comparison (AIC) for BTC and ETH", fontsize=14)
fig.tight_layout()

out_path = os.path.join(FIG_DIR, "model_comparison_aicbic.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")