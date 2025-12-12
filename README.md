# 🚀 Customer Lifetime Prediction Engine

A modular, end-to-end Machine Learning pipeline designed to predict customer tenure and lifetime value. Unlike standard churn classification, this system uses regression analysis to estimate **exactly how long** a customer will remain loyal to the platform.

---

## 🎯 Project Goal
**"We don't just predict IF they will leave; we predict WHEN."**
By analyzing demographics, usage frequency, and support interactions, we forecast the total duration (in months) a customer represents value to the company.

---

## 🏗 Architecture & Modules

### 1. Data Ingestion Layer (`/src/ingestion`)
* **Function:** Loads raw customer data from CSV/Excel sources.
* **Logic:** Filters dataset to train *only* on historical churned customers (to learn completed lifecycles), ensuring accurate "Total Lifetime" modeling.

### 2. Preprocessing Module (`/src/preprocessing`)
* **Cleaning:** Handles missing values and removes statistical outliers.
* **Encoding:**
    * Label Encoding: `Gender`, `Subscription Type`
    * One-Hot Encoding: `Contract Length`
* **Scaling:** Standardizes financial features (`Total Spend`) using `StandardScaler`.

### 3. Feature Engineering (`/src/features`)
* **Correlation Analysis:** Removes highly correlated features (multicollinearity check).
* **Feature Selection:** Drops non-predictive columns like `CustomerID`.
* **New Features:** Calculates `Average Spend per Month` to normalize spending behavior.

### 4. Model Training (`/src/training`)
We train multiple regression algorithms to predict **Tenure (Months)**:
* **Linear Regression:** Baseline model for interpretability.
* **Random Forest Regressor:** Captures non-linear relationships.
* **XGBoost Regressor:** High-performance gradient boosting for maximum accuracy.
* **Optimization:** Hyperparameter tuning via `GridSearchCV`.

### 5. Evaluation (`/src/evaluation`)
Performance is measured using regression metrics:
* **RMSE (Root Mean Squared Error):** The average error in months.
* **R² Score:** How well our model explains the variance in customer lifespan.
* **Visuals:** Residual plots to check for bias.

### 6. Deployment (`/src/deployment`)
* **FastAPI Backend:** Endpoint `/predict_lifetime` accepts customer attributes and returns predicted tenure in months.
* **Streamlit Dashboard:** Interactive tool for business stakeholders to input customer details and see the predicted lifespan graph.

---

## 🛠 Tech Stack

* **Language:** Python 3.9+
* **Data:** Pandas, NumPy
* **ML Core:** Linear regression, Random Forest, XGBoost
* **API:** FastAPI, Uvicorn
* **UI:** Streamlit

---

## 🏃 Usage

### Training
```bash
python main.py --mode train --data_path data/customer_data.csv
