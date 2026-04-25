# 🏠 House Price Prediction

A Machine Learning project that predicts house prices based on various features like area, location, number of rooms, and amenities.

---

## 📌 Project Overview

| Field | Details |
|-------|---------|
| **Type** | Supervised Learning — Regression |
| **Domain** | Real Estate |
| **Best Model** | Lasso Regression |
| **Best R² Score** | 0.9512 (95.12% accuracy) |
| **Dataset Size** | 1,500 houses |

---

## 🎯 Objective

To build a Machine Learning model that can **predict the selling price of a house** based on features like square footage, number of bedrooms, neighborhood, house age, and more.

This is similar to how platforms like **Zillow or MagicBricks** estimate property prices.

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `sqft_living` | Living area in square feet |
| `sqft_lot` | Total lot size in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `floors` | Number of floors |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of last renovation (0 = never) |
| `garage` | Has garage? (1 = Yes, 0 = No) |
| `pool` | Has pool? (1 = Yes, 0 = No) |
| `waterfront` | Waterfront property? (1 = Yes, 0 = No) |
| `neighborhood` | Area type (Downtown, Suburb, Rural, etc.) |
| `house_type` | Type of house (Detached, Flat, etc.) |
| `condition` | Condition rating (1–5) |
| `grade` | Construction quality grade (3–13) |
| `price` | **Target variable** — Sale price in USD |

---

## ⚙️ Project Pipeline

```
Data Generation → Preprocessing → EDA → Feature Engineering → Model Training → Evaluation
```

### Step 1 — Data Collection
- Generated a realistic dataset of **1,500 houses**
- Price range: **$222,871 – $1,163,501**
- Intentionally added 2% missing values to simulate real-world data

### Step 2 — Preprocessing
- Filled missing values using **median and mode imputation**
- Applied **Label Encoding** on categorical columns (neighborhood, house type)

### Step 3 — Feature Engineering
Created 4 new features from existing data:
- `house_age` = 2023 − year built
- `was_renovated` = 1 if renovated, else 0
- `bed_bath_ratio` = bedrooms ÷ bathrooms
- `total_sqft` = living area + lot area

### Step 4 — Exploratory Data Analysis (EDA)
Generated 8 visualizations including:
- Price distribution histogram
- Price vs Living Area scatter plot
- Average price by neighborhood
- Correlation heatmap
- Price by number of bedrooms
- Waterfront premium analysis

### Step 5 — Model Training
Applied **log transformation** on price (target variable) for better model fit.
Trained and compared 5 regression models:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|---------|
| Linear Regression | $39,242 | $30,442 | 0.9495 |
| Ridge Regression | $38,884 | $30,153 | 0.9504 |
| **Lasso Regression** | **$38,591** | **$29,884** | **0.9512** ✅ |
| Random Forest | $47,622 | $37,474 | 0.9257 |
| Gradient Boosting | $40,127 | $31,591 | 0.9472 |

### Step 6 — Results
- Best model: **Lasso Regression**
- R² Score: **0.9512** — model explains 95% of price variation
- Lasso won because it performed automatic feature selection

---

## 📈 Output Charts

| File | Description |
|------|-------------|
| `outputs/01_eda.png` | 8-panel Exploratory Data Analysis |
| `outputs/02_model_results.png` | Model comparison, actual vs predicted, residuals, feature importance |

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — ML models, preprocessing, metrics
- **matplotlib** — plotting
- **seaborn** — statistical visualizations

---

## ▶️ How to Run

**Step 1 — Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Step 2 — Run the script:**
```bash
python house_price_prediction.py
```

**Step 3 — View outputs:**
Charts are saved in the `outputs/` folder automatically.

---

## 📁 Project Structure

```
house_price/
├── house_price_prediction.py    ← Main Python script
├── README.md                    ← This file
└── outputs/
    ├── 01_eda.png               ← EDA visualizations
    └── 02_model_results.png     ← Model result charts
```

---

## 💡 Key Learnings

- How to handle missing data using imputation
- Importance of feature engineering in improving model performance
- Log transformation helps regression models on skewed targets
- Lasso regression performs built-in feature selection via L1 regularization
- R² score is the primary metric for regression problems

---

## 👨‍💻 Author

**SPANDU**
B.Tech — Artificial Intelligence & Machine Learning
PES University, Bengaluru
