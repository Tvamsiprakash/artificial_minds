# 🚀 Customer Churn Prediction - Regression Model for Predictive Analysis

# Sprint 0 - Design Flow:

A modular, end-to-end Machine Learning pipeline designed to predict customer tenure and lifetime value. Unlike standard churn classification, this system uses regression analysis to estimate **exactly how long** a customer will remain loyal to the platform.

---

## 🎯 Project Goal
**"We don't just predict IF they will leave; we predict WHEN."**
By analyzing demographics, usage frequency, and support interactions, we forecast the total duration (in days) a customer represents value to the company.

---

## 🏗 Architecture & Modules

### 1. Data Ingestion Layer 
* **Function:** Loads raw customer data from CSV/Excel sources.
* **Logic:** Filters dataset to train *only* on historical churned customers (to learn completed lifecycles), ensuring accurate "Total Lifetime" modeling.

### 2. Preprocessing Module
* **Cleaning:** Handles missing values and removes statistical outliers.
* **Encoding:**
    * Label Encoding: `Gender`, `Subscription Type`
    * One-Hot Encoding: `Contract Length`
* **Scaling:** Standardizes financial features (`Total Spend`) using `StandardScaler`.

### 3. Feature Engineering 
* **Correlation Analysis:** Removes highly correlated features (multicollinearity check).
* **Feature Selection:** Drops non-predictive columns like `CustomerID`.
* **New Features:** Calculates `Average Spend per Month` to normalize spending behavior.

### 4. Model Training 
We train multiple regression algorithms to predict **Tenure (Months)**:
* **Linear Regression:** Baseline model for interpretability.
* **Random Forest Regressor:** Captures non-linear relationships.
* **XGBoost Regressor:** High-performance gradient boosting for maximum accuracy.
* **Optimization:** Hyperparameter tuning via `GridSearchCV`.

### 5. Evaluation
Performance is measured using regression metrics:
* **RMSE (Root Mean Squared Error):** The average error in months.
* **R² Score:** How well our model explains the variance in customer lifespan.
* **Visuals:** Residual plots to check for bias.

### 6. Deployment
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
# Sprint 1 & 2 - Model implementation and Deployment



This project focuses on predicting customer tenure using machine learning models. The dataset contains customer information, and the target variable is **Tenure**. Multiple regression models were trained, evaluated, and compared to identify the best-performing approach.

The final **tuned XGBoost model** achieved the highest accuracy and was saved for deployment.

---

## ⚙️ Workflow

### 1. Data Preparation and Exploratory Data Analysis
* **Filtered Dataset:** Retained only customers with Churn = 1 to ensure Tenure represents complete customer lifecycle.
* **Data Quality Check:** Examined missing values, duplicates, dataset structure, and summary statistics.
* **Exploratory Data Analysis:** Generated histograms for numeric features and countplots for categorical features to understand distributions and class balance.
* **Outlier Handling:** Applied IQR-based capping to numeric features to reduce the influence of extreme values.
* **Feature Encoding & Scaling:** Label-encoded categorical variables and standardized numeric features for uniform model input.
* **Correlation Analysis:** Produced a correlation heatmap after removing ID and Churn columns to identify relationships between variables.
<img width="1210" height="945" alt="image" src="https://github.com/user-attachments/assets/5f8c3f36-d6c4-431d-addf-ba4768214dc8" />


### 2. Models Implemented
* Linear Regression
* Random Forest Regressor
* XGBoost Regressor (Initial)
* **XGBoost Regressor (Tuned with GridSearchCV)**

### 3. Evaluation Metrics
* **Normalized MAE:** Mean Absolute Error relative to data range.
* **Normalized RMSE:** Root Mean Squared Error relative to data range.
* **R² Score:** Goodness of fit (higher is better).

---

## 📈 Results


| Model | R² Score | Performance Verdict |
| :--- | :--- | :--- |
| Linear Regression | 0.1100 | Poor (Underfitting) |
| Random Forest | 0.6581 | Good |
| XGBoost (Initial) | 0.7568 | Very Good |
| **XGBoost (Tuned)** | **0.8148** | **Best Performing** |

➡️ **Best Model:** Tuned XGBoost Regressor

---
## some reference for Model evalulation metric scores:
# linear Regression

<img width="406" height="76" alt="Screenshot 2025-12-12 175035" src="https://github.com/user-attachments/assets/58759485-3f6e-4d97-b1da-a8c8acd80c83" />

# Random Forest
<img width="456" height="89" alt="Screenshot 2025-12-12 174956" src="https://github.com/user-attachments/assets/44c070a4-7c93-47f8-8151-430f47bf71c9" />


# XgBoost
<img width="527" height="91" alt="Screenshot 2025-12-12 174948" src="https://github.com/user-attachments/assets/247f754a-9e98-420f-aee2-ffa85475dc7f" />

# Xgboost tuned - hyperparameter Tuning
<img width="1039" height="199" alt="Screenshot 2025-12-12 174929" src="https://github.com/user-attachments/assets/fb34e9ca-05d9-4978-839a-dcdcecd282c9" />




## 📊 Visualization

A bar chart was created to compare the R² scores of all models.

<img width="984" height="583" alt="download" src="https://github.com/user-attachments/assets/dc7973ed-61a0-4121-9f73-2c8c0cd5beb5" />


* **Darker bars** indicate better performance.
* **Tuned XGBoost** clearly outperformed other models, capturing non-linear patterns effectively.

---

## 💾 Model Saving

The best model was serialized using `pickle` for future use:

```python
import pickle

with open("best_xgboost_tuned.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

