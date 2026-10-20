# 🛍️ Retail Sales Analytics & Prediction System

An end-to-end real-world Data Science and Business Intelligence application built with **Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, and SQLite**.

---

## 🌟 Key Features

1. **Data Import & Validation:** Supports uploading custom retail CSV datasets or automatically generates a realistic 3,000+ transaction synthetic dataset with missing values and duplicates for hygiene testing.
2. **Data Cleaning & Hygiene Engine:** Validates schema, removes duplicates, handles missing values, clips outliers, converts dates, and engineers 11 derived features (e.g. `Year_Month`, `Gross_Amount`, `Age_Group`).
3. **Exploratory Data Analysis (EDA):** Interactive Plotly visualizations for temporal trends, category revenue breakdown, top products, store location performance, payment distribution, customer age/gender demographics, and discount vs sales impact.
4. **Executive Business KPI Dashboard:** High-level metric cards for Total Revenue, Total Orders, Active Customers, Average Order Value (AOV), Total Units Sold, Best Category, Best Product, and MoM Growth.
5. **Customer RFM Segmentation:** Evaluates Recency, Frequency, and Monetary metrics and uses **K-Means Clustering** to segment customers into:
   - High Value Customers
   - Regular / Loyal Customers
   - New / Potential Customers
   - At-Risk / Inactive Customers
6. **Machine Learning Sales Forecasting:** Trains and compares **Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor** models on time-series daily sales with lag (1, 7, 14, 30 days) and rolling window features. Evaluates performance using MAE, MSE, RMSE, and R² Score, and forecasts future 30-day sales.
7. **Automated Business Insights:** Synthesizes actionable recommendations for marketing, supply chain, store management, and customer churn prevention.
8. **SQLite Integration:** Stores raw & cleaned transactions in SQLite (`sales_analytics.db`) and allows executing live custom SQL queries.

---

## 📂 Project Directory Structure

```text
retail_sales_project/
├── data/
│   ├── raw_retail_sales.csv         # Synthetic/raw transaction data
│   └── cleaned_retail_sales.csv     # Preprocessed dataset
├── database/
│   └── sales_analytics.db           # SQLite database
├── notebooks/
│   └── eda_and_modeling.ipynb       # Complete Jupyter notebook for EDA & ML
├── src/
│   ├── __init__.py
│   ├── data_generator.py            # Synthetic retail dataset generator
│   ├── data_cleaner.py              # Data cleaning & feature engineering
│   ├── database.py                  # SQLite database wrapper & SQL queries
│   ├── eda.py                       # Plotly visualization functions
│   ├── segmentation.py              # RFM Analysis & K-Means clustering
│   ├── predictor.py                 # Time-series feature engineering & ML models
│   ├── insights.py                  # Automated Business Insights engine
│   ├── generate_report.py           # Markdown report exporter
│   └── make_notebook.py             # Jupyter notebook maker
├── models/
│   ├── rfm_kmeans.pkl               # Saved RFM K-Means model & scaler
│   └── sales_forecaster.pkl         # Saved best sales prediction model
├── outputs/
│   └── summary_report.md            # Generated Markdown executive summary
├── app.py                           # Main Streamlit dashboard application
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation & setup guide
```

---

## 🚀 Setup & Installation Instructions

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Navigate to Project Directory
```bash
cd retail_sales_project
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Quick Start Commands

### A. Run Data Pipeline & Train Models
You can run individual modules or train the models directly via CLI:

```bash
# 1. Generate Synthetic Dataset
python src/data_generator.py

# 2. Clean Data & Save Cleaned Dataset
python src/data_cleaner.py

# 3. Store Data in SQLite Database
python src/database.py

# 4. Train RFM Customer Segmentation Model
python src/segmentation.py

# 5. Train & Evaluate ML Sales Forecasting Models
python src/predictor.py

# 6. Generate Executive Summary Report
python src/generate_report.py
```

### B. Run the Interactive Streamlit Web App
Launch the interactive web application in your browser:

```bash
streamlit run app.py
```

---

## 📓 Running in VS Code & Jupyter Notebooks

1. Open VS Code:
   ```bash
   code .
   ```
2. Open `notebooks/eda_and_modeling.ipynb`.
3. Select your Python kernel and click **Run All** to execute the complete EDA and ML modeling workflow interactively.

---

## 📊 Summary Report Excerpt
An automated executive summary report is compiled to `outputs/summary_report.md` after running the pipeline, containing total metrics, cluster counts, model evaluation tables, and strategic recommendations.
