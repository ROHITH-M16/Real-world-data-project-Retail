"""
Report Generator Module.
Generates an executive markdown summary report in outputs/summary_report.md.
"""

import os
import pandas as pd
import numpy as np
from data_cleaner import clean_retail_data
from segmentation import calculate_rfm, train_rfm_kmeans
from predictor import prepare_time_series_dataset, train_and_evaluate_models
from insights import generate_business_insights

def generate_markdown_report(output_path=None):
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_retail_sales.csv")
    if not os.path.exists(raw_path):
        from data_generator import generate_retail_data
        df_raw = generate_retail_data(3200, output_path=raw_path)
    else:
        df_raw = pd.read_csv(raw_path)
        
    df_clean, cleaning_summary = clean_retail_data(df_raw)
    
    rfm_raw = calculate_rfm(df_clean)
    rfm_seg, profile, _ = train_rfm_kmeans(rfm_raw, n_clusters=4)
    
    ts_df = prepare_time_series_dataset(df_clean)
    eval_res = train_and_evaluate_models(ts_df)
    
    insights = generate_business_insights(df_clean, rfm_seg)
    
    total_rev = df_clean['Total Amount'].sum()
    total_orders = len(df_clean)
    total_custs = df_clean['Customer ID'].nunique()
    aov = total_rev / total_orders
    
    seg_counts = rfm_seg['Customer_Segment'].value_counts()
    
    metrics_table_md = eval_res['metrics_df'].to_markdown(index=False)
    
    report_md = f"""# 🛍️ Retail Sales Analytics & Prediction Executive Summary

**Generated On:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Project:** Retail Sales Analytics & Prediction System  

---

## 📌 Executive Business KPIs

| Metric | Value |
| :--- | :--- |
| **Total Revenue** | **${total_rev:,.2f}** |
| **Total Transactions** | **{total_orders:,}** |
| **Unique Customers** | **{total_custs:,}** |
| **Average Order Value (AOV)** | **${aov:,.2f}** |
| **Total Units Sold** | **{df_clean['Quantity'].sum():,}** |

---

## 🎯 Customer Segmentation (RFM Analysis)

Customers were grouped into 4 strategic tiers based on Recency, Frequency, and Monetary scores using K-Means Clustering:

- 🟢 **High Value Customers:** {seg_counts.get('High Value Customers', 0):,} customers
- 🔵 **Regular / Loyal Customers:** {seg_counts.get('Regular / Loyal Customers', 0):,} customers
- 🟡 **New / Potential Customers:** {seg_counts.get('New / Potential Customers', 0):,} customers
- 🔴 **At-Risk / Inactive Customers:** {seg_counts.get('At-Risk / Inactive Customers', 0):,} customers

---

## 🤖 Machine Learning Sales Forecasting

Supervised machine learning algorithms were trained on daily sales time series with lag and rolling features:

{metrics_table_md}

**Best Performing Model:** `{eval_res['best_model_name']}`

---

## 💡 Key Business Insights & Actionable Recommendations

### Key Findings
"""
    for ins in insights['insights']:
        report_md += f"- **{ins['icon']} {ins['title']}:** {ins['text']}\n"
        
    report_md += "\n### Strategic Recommendations\n"
    for rec in insights['recommendations']:
        report_md += f"- **{rec['icon']} {rec['title']} [{rec['area']}]:** {rec['action']}\n"
        
    report_md += "\n---\n*Report generated automatically by Retail Sales Analytics & Prediction System.*"
    
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "summary_report.md")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Summary report generated at: {output_path}")

if __name__ == "__main__":
    generate_markdown_report()
