"""
Business Insights and Automated Recommendations Module.
Analyzes transaction data, RFM segments, and store performance to generate actionable business intelligence.
"""

import pandas as pd
import numpy as np

def generate_business_insights(df, rfm_df=None):
    """
    Analyzes retail dataframe and returns a dictionary of insights & actionable recommendations.
    """
    insights = []
    recommendations = []
    
    # 1. Top Revenue Category & Share
    cat_summary = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    top_cat = cat_summary.index[0]
    top_cat_rev = cat_summary.iloc[0]
    total_rev = df['Total Amount'].sum()
    top_cat_pct = (top_cat_rev / total_rev) * 100
    
    insights.append({
        'title': 'Highest Revenue Category',
        'category': 'Category Analysis',
        'text': f"**{top_cat}** is the leading product category, contributing **${top_cat_rev:,.2f}** ({top_cat_pct:.1f}%) of total business revenue.",
        'icon': '🏆',
        'status': 'success'
    })
    
    # 2. Products to Promote (High Unit Price but Lower Quantity Sold)
    prod_summary = df.groupby(['Product Name', 'Product Category']).agg(
        Total_Rev=('Total Amount', 'sum'),
        Units_Sold=('Quantity', 'sum'),
        Avg_Price=('Unit Price', 'mean')
    ).reset_index()
    
    # High price, moderate/low volume -> ideal for promotion
    high_value_prods = prod_summary.sort_values(by=['Avg_Price', 'Units_Sold'], ascending=[False, True])
    promo_candidates = high_value_prods.head(3)['Product Name'].tolist()
    
    recommendations.append({
        'title': 'Targeted Promotional Campaign',
        'area': 'Marketing & Sales',
        'action': f"Run targeted promotional discount campaigns for high-margin products: **{', '.join(promo_candidates)}** to boost volume.",
        'icon': '🎯'
    })
    
    # 3. Store Performance & Improvement Needs
    store_summary = df.groupby('Store Location')['Total Amount'].sum().sort_values(ascending=False)
    best_store = store_summary.index[0]
    worst_store = store_summary.index[-1]
    worst_store_rev = store_summary.iloc[-1]
    best_store_rev = store_summary.iloc[0]
    
    insights.append({
        'title': 'Store Performance Contrast',
        'category': 'Store Location',
        'text': f"**{best_store}** leads sales at **${best_store_rev:,.2f}**, while **{worst_store}** lags behind at **${worst_store_rev:,.2f}**.",
        'icon': '🏪',
        'status': 'warning'
    })
    
    recommendations.append({
        'title': 'Underperforming Store Optimization',
        'area': 'Store Operations',
        'action': f"Review inventory stock, local marketing, and staffing for **{worst_store}** to narrow the {((best_store_rev - worst_store_rev)/worst_store_rev)*100:.0f}% performance gap.",
        'icon': '🛠️'
    })
    
    # 4. Monthly Seasonality (Peak vs Trough)
    monthly_sales = df.groupby('Month_Name')['Total Amount'].sum()
    # Sort by month order if possible
    peak_month = monthly_sales.idxmax()
    peak_rev = monthly_sales.max()
    trough_month = monthly_sales.idxmin()
    trough_rev = monthly_sales.min()
    
    insights.append({
        'title': 'Sales Seasonality Peak & Trough',
        'category': 'Temporal Trends',
        'text': f"Peak revenue occurs in **{peak_month}** (${peak_rev:,.2f}), whereas **{trough_month}** experiences lower sales volume (${trough_rev:,.2f}).",
        'icon': '📅',
        'status': 'info'
    })
    
    recommendations.append({
        'title': 'Inventory & Supply Chain Timing',
        'area': 'Supply Chain',
        'action': f"Increase inventory orders 3 weeks prior to **{peak_month}** to prevent stockouts, and launch flash sales during **{trough_month}**.",
        'icon': '📦'
    })
    
    # 5. Customer Segmentation At-Risk Alert
    if rfm_df is not None and 'Customer_Segment' in rfm_df.columns:
        at_risk_count = (rfm_df['Customer_Segment'] == 'At-Risk / Inactive Customers').sum()
        total_custs = len(rfm_df)
        at_risk_pct = (at_risk_count / total_custs) * 100
        
        insights.append({
            'title': 'At-Risk Customer Churn Warning',
            'category': 'Customer Retention',
            'text': f"**{at_risk_count} customers** ({at_risk_pct:.1f}% of customer base) show high recency & low purchase frequency, indicating imminent churn risk.",
            'icon': '⚠️',
            'status': 'danger'
        })
        
        recommendations.append({
            'title': 'Automated Win-Back Campaign',
            'area': 'Customer Relationship Management',
            'action': "Deploy automated re-engagement email campaigns with personalized 15% discount vouchers to At-Risk customers.",
            'icon': '✉️'
        })
        
    return {
        'insights': insights,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    import os
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        res = generate_business_insights(df)
        print("Generated Insights:")
        for ins in res['insights']:
            print(f"- {ins['title']}: {ins['text']}")
