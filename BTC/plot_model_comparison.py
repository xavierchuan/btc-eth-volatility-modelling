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
# Plot AIC comparison (zoomed)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

colors = {"GJR-GARCH(1,1)": "#1f77b4",
          "EGARCH(1,1)": "#ff7f0e",
          "GARCH(1,1)": "#2ca02c"}

for ax, asset in zip(axes, ["BTC", "ETH"]):
    subset = df[df["Asset"] == asset]
    ax.bar(subset["Model"], subset["AIC"], 
           color=[colors[m] for m in subset["Model"]])
    ax.set_title(f"{asset} - AIC Comparison", fontsize=11)
    ax.set_ylabel("AIC")
    ax.tick_params(axis="x", rotation=30)

    # Zoom in Y-axis so differences are visible
    aic_min, aic_max = subset["AIC"].min(), subset["AIC"].max()
    margin = (aic_max - aic_min) * 0.2  # 20% margin for better visibility
    ax.set_ylim(aic_min - margin, aic_max + margin)

    # Highlight the best model
    best_model = subset.loc[subset["AIC"].idxmin(), "Model"]
    ax.axhline(y=subset["AIC"].min(), color="red", linestyle="--", linewidth=1)
    ax.text(1, subset["AIC"].min() + margin * 0.1, 
            f"Best: {best_model}", 
            color="red", fontsize=9, ha="center")

fig.suptitle("Model Comparison (AIC) for BTC and ETH (Zoomed)", fontsize=14)
fig.tight_layout()

out_path = os.path.join(FIG_DIR, "model_comparison_aicbic.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")