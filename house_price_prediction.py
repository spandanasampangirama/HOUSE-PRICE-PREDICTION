

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = ["#4361EE", "#3F37C9", "#4895EF", "#4CC9F0", "#F72585"]

# ─────────────────────────────────────────
# STEP 1: GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: DATA COLLECTION & GENERATION")
print("="*60)

np.random.seed(42)
n = 1500

neighborhoods = ["Downtown", "Suburb", "Rural", "Uptown", "Midtown"]
house_types   = ["Detached", "Semi-Detached", "Terraced", "Flat"]

df = pd.DataFrame({
    "sqft_living"   : np.random.randint(500, 5000, n),
    "sqft_lot"      : np.random.randint(1000, 15000, n),
    "bedrooms"      : np.random.randint(1, 7, n),
    "bathrooms"     : np.random.randint(1, 5, n),
    "floors"        : np.random.choice([1, 1.5, 2, 2.5, 3], n),
    "yr_built"      : np.random.randint(1950, 2023, n),
    "yr_renovated"  : np.random.choice([0]*8 + list(range(1990, 2023)), n),
    "garage"        : np.random.choice([0, 1], n, p=[0.3, 0.7]),
    "pool"          : np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "neighborhood"  : np.random.choice(neighborhoods, n),
    "house_type"    : np.random.choice(house_types, n),
    "condition"     : np.random.randint(1, 6, n),          # 1-5
    "grade"         : np.random.randint(3, 14, n),         # 3-13
    "waterfront"    : np.random.choice([0, 1], n, p=[0.9, 0.1]),
})

# Realistic price formula
neigh_factor = {"Downtown":1.4, "Suburb":1.0, "Rural":0.7, "Uptown":1.6, "Midtown":1.2}
df["price"] = (
      df["sqft_living"] * 120
    + df["sqft_lot"] * 5
    + df["bedrooms"] * 8000
    + df["bathrooms"] * 12000
    + df["floors"] * 15000
    + df["condition"] * 10000
    + df["grade"] * 20000
    + df["garage"] * 25000
    + df["pool"] * 40000
    + df["waterfront"] * 100000
    + df["neighborhood"].map(neigh_factor) * 30000
    + (2023 - df["yr_built"]) * (-300)
    + np.random.normal(0, 30000, n)
).clip(50000, 2000000).astype(int)

# Inject 2% missing values for realism
for col in ["sqft_lot", "yr_renovated", "condition"]:
    df.loc[df.sample(frac=0.02).index, col] = np.nan

print(f"  Dataset shape : {df.shape}")
print(f"  Price range   : ${df['price'].min():,} – ${df['price'].max():,}")
print(f"  Avg price     : ${df['price'].mean():,.0f}")
print(f"  Missing vals  : {df.isnull().sum().sum()} cells")

# ─────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: DATA PREPROCESSING")
print("="*60)

df["sqft_lot"].fillna(df["sqft_lot"].median(), inplace=True)
df["yr_renovated"].fillna(0, inplace=True)
df["condition"].fillna(df["condition"].mode()[0], inplace=True)

# Feature engineering
df["house_age"]      = 2023 - df["yr_built"]
df["was_renovated"]  = (df["yr_renovated"] > 0).astype(int)
df["price_per_sqft"] = (df["price"] / df["sqft_living"]).round(2)
df["bed_bath_ratio"] = (df["bedrooms"] / df["bathrooms"].replace(0, 0.5)).round(2)
df["total_sqft"]     = df["sqft_living"] + df["sqft_lot"]

# Encode categoricals
le = LabelEncoder()
df["neighborhood_enc"] = le.fit_transform(df["neighborhood"])
df["house_type_enc"]   = le.fit_transform(df["house_type"])

print("  ✓ Missing values filled")
print("  ✓ Feature engineering: house_age, was_renovated, price_per_sqft, bed_bath_ratio")
print("  ✓ Label encoding applied")

# ─────────────────────────────────────────
# STEP 3: EDA VISUALIZATIONS
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

fig = plt.figure(figsize=(20, 16))
fig.suptitle("House Price Prediction — Exploratory Data Analysis", fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Price distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["price"]/1e6, bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
ax1.set_title("Price Distribution", fontweight="bold")
ax1.set_xlabel("Price (Millions $)")
ax1.set_ylabel("Count")

# Price vs sqft
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(df["sqft_living"], df["price"]/1e6, alpha=0.3, color=PALETTE[2], s=12)
ax2.set_title("Price vs Living Area", fontweight="bold")
ax2.set_xlabel("Sq Ft Living")
ax2.set_ylabel("Price (M$)")

# Price by neighborhood
ax3 = fig.add_subplot(gs[0, 2])
neigh_avg = df.groupby("neighborhood")["price"].mean().sort_values(ascending=False)
bars = ax3.bar(neigh_avg.index, neigh_avg.values/1e3, color=PALETTE[:len(neigh_avg)])
ax3.set_title("Avg Price by Neighborhood", fontweight="bold")
ax3.set_ylabel("Avg Price (K$)")
ax3.tick_params(axis="x", rotation=30)

# Correlation heatmap
ax4 = fig.add_subplot(gs[1, :2])
num_cols = ["price","sqft_living","bedrooms","bathrooms","grade","condition","house_age","total_sqft"]
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax4, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size":8})
ax4.set_title("Feature Correlation Heatmap", fontweight="bold")

# Boxplot: price by bedrooms
ax5 = fig.add_subplot(gs[1, 2])
bed_groups = [df[df["bedrooms"]==b]["price"].values/1e3 for b in sorted(df["bedrooms"].unique())]
ax5.boxplot(bed_groups, labels=sorted(df["bedrooms"].unique()), patch_artist=True,
            boxprops=dict(facecolor=PALETTE[3], alpha=0.6))
ax5.set_title("Price by Bedrooms", fontweight="bold")
ax5.set_xlabel("Bedrooms")
ax5.set_ylabel("Price (K$)")

# Price vs grade
ax6 = fig.add_subplot(gs[2, 0])
grade_avg = df.groupby("grade")["price"].mean()
ax6.plot(grade_avg.index, grade_avg.values/1e3, marker="o", color=PALETTE[4], linewidth=2)
ax6.set_title("Avg Price by Grade", fontweight="bold")
ax6.set_xlabel("Grade")
ax6.set_ylabel("Avg Price (K$)")

# Waterfront effect
ax7 = fig.add_subplot(gs[2, 1])
wf_data = [df[df["waterfront"]==0]["price"]/1e3, df[df["waterfront"]==1]["price"]/1e3]
ax7.boxplot(wf_data, labels=["No Waterfront","Waterfront"], patch_artist=True,
            boxprops=dict(facecolor=PALETTE[1], alpha=0.6))
ax7.set_title("Waterfront Premium", fontweight="bold")
ax7.set_ylabel("Price (K$)")

# House age vs price
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(df["house_age"], df["price"]/1e3, alpha=0.2, color=PALETTE[0], s=8)
ax8.set_title("House Age vs Price", fontweight="bold")
ax8.set_xlabel("Age (Years)")
ax8.set_ylabel("Price (K$)")

plt.savefig("outputs/01_eda.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
plt.close()
print("  ✓ EDA charts saved → outputs/01_eda.png")

# ─────────────────────────────────────────
# STEP 4: MODEL TRAINING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: MODEL TRAINING & EVALUATION")
print("="*60)

features = ["sqft_living","sqft_lot","bedrooms","bathrooms","floors",
            "grade","condition","garage","pool","waterfront",
            "neighborhood_enc","house_type_enc","house_age","was_renovated",
            "bed_bath_ratio","total_sqft"]

X = df[features]
y = np.log1p(df["price"])

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=10),
    "Lasso Regression"  : Lasso(alpha=0.001),
    "Random Forest"     : RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}

results = {}
print(f"\n  {'Model':<22} {'RMSE ($)':>12} {'MAE ($)':>12} {'R² Score':>10}")
print("  " + "-"*58)

for name, model in models.items():
    X_tr = X_train_s if "Regression" in name else X_train
    X_te = X_test_s  if "Regression" in name else X_test
    model.fit(X_tr, y_train)
    preds = np.expm1(model.predict(X_te))
    actuals = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    r2   = r2_score(actuals, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "model": model,
                     "preds": preds, "actuals": actuals}
    print(f"  {name:<22} ${rmse:>11,.0f} ${mae:>11,.0f} {r2:>10.4f}")

best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n  ✓ Best model: {best_model_name}  (R² = {results[best_model_name]['R2']:.4f})")

# ─────────────────────────────────────────
# STEP 5: RESULT VISUALIZATIONS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("House Price Prediction — Model Results", fontsize=16, fontweight="bold")

# Model comparison
ax = axes[0, 0]
names = list(results.keys())
r2s   = [results[n]["R2"] for n in names]
colors = [PALETTE[4] if n == best_model_name else PALETTE[2] for n in names]
bars = ax.barh(names, r2s, color=colors, edgecolor="white")
ax.set_title("R² Score Comparison", fontweight="bold")
ax.set_xlabel("R² Score")
ax.set_xlim(0, 1)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# Actual vs predicted (best model)
ax = axes[0, 1]
best = results[best_model_name]
ax.scatter(best["actuals"]/1e3, best["preds"]/1e3, alpha=0.3, color=PALETTE[0], s=12)
lims = [0, max(best["actuals"].max(), best["preds"].max())/1e3]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
ax.set_title(f"Actual vs Predicted ({best_model_name})", fontweight="bold")
ax.set_xlabel("Actual Price (K$)")
ax.set_ylabel("Predicted Price (K$)")
ax.legend()

# Residuals
ax = axes[1, 0]
residuals = best["actuals"] - best["preds"]
ax.scatter(best["preds"]/1e3, residuals/1e3, alpha=0.3, color=PALETTE[3], s=12)
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_title("Residuals Plot", fontweight="bold")
ax.set_xlabel("Predicted Price (K$)")
ax.set_ylabel("Residual (K$)")

# Feature importance (Random Forest)
ax = axes[1, 1]
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True).tail(10)
importances.plot(kind="barh", ax=ax, color=PALETTE[1])
ax.set_title("Top 10 Feature Importances (RF)", fontweight="bold")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("outputs/02_model_results.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
plt.close()



