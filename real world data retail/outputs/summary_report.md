# 🛍️ Retail Sales Analytics & Prediction Executive Summary

**Generated On:** 2026-10-08 18:20:44  
**Project:** Retail Sales Analytics & Prediction System  

---

## 📌 Executive Business KPIs

| Metric | Value |
| :--- | :--- |
| **Total Revenue** | **$1,328,931.90** |
| **Total Transactions** | **3,200** |
| **Unique Customers** | **399** |
| **Average Order Value (AOV)** | **$415.29** |
| **Total Units Sold** | **15,186** |

---

## 🎯 Customer Segmentation (RFM Analysis)

Customers were grouped into 4 strategic tiers based on Recency, Frequency, and Monetary scores using K-Means Clustering:

- 🟢 **High Value Customers:** 139 customers
- 🔵 **Regular / Loyal Customers:** 81 customers
- 🟡 **New / Potential Customers:** 140 customers
- 🔴 **At-Risk / Inactive Customers:** 39 customers

---

## 🤖 Machine Learning Sales Forecasting

Supervised machine learning algorithms were trained on daily sales time series with lag and rolling features:

| Model             |   MAE ($) |         MSE |   RMSE ($) |   R² Score |
|:------------------|----------:|------------:|-----------:|-----------:|
| Linear Regression |    804.56 | 1.02738e+06 |    1013.6  |    -0.0085 |
| Gradient Boosting |    828.38 | 1.14964e+06 |    1072.21 |    -0.1285 |
| Random Forest     |    841.82 | 1.16312e+06 |    1078.48 |    -0.1417 |

**Best Performing Model:** `Linear Regression`

---

## 💡 Key Business Insights & Actionable Recommendations

### Key Findings
- **🏆 Highest Revenue Category:** **Electronics** is the leading product category, contributing **$456,000.93** (34.3%) of total business revenue.
- **🏪 Store Performance Contrast:** **Online Store** leads sales at **$214,879.99**, while **New York** lags behind at **$171,786.51**.
- **📅 Sales Seasonality Peak & Trough:** Peak revenue occurs in **Jan** ($137,989.30), whereas **Oct** experiences lower sales volume ($67,506.79).
- **⚠️ At-Risk Customer Churn Warning:** **39 customers** (9.8% of customer base) show high recency & low purchase frequency, indicating imminent churn risk.

### Strategic Recommendations
- **🎯 Targeted Promotional Campaign [Marketing & Sales]:** Run targeted promotional discount campaigns for high-margin products: **4K Monitor, Smart Watch, Vacuum Cleaner** to boost volume.
- **🛠️ Underperforming Store Optimization [Store Operations]:** Review inventory stock, local marketing, and staffing for **New York** to narrow the 25% performance gap.
- **📦 Inventory & Supply Chain Timing [Supply Chain]:** Increase inventory orders 3 weeks prior to **Jan** to prevent stockouts, and launch flash sales during **Oct**.
- **✉️ Automated Win-Back Campaign [Customer Relationship Management]:** Deploy automated re-engagement email campaigns with personalized 15% discount vouchers to At-Risk customers.

---
*Report generated automatically by Retail Sales Analytics & Prediction System.*