"""
Retail Sales Analytics & Prediction System - Streamlit Dashboard Entry Point.
Provides an interactive multi-tab dashboard for KPI analytics, EDA, RFM segmentation, ML sales forecasting,
business insights, and SQLite database queries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
from datetime import datetime

# Add src module directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_generator import generate_retail_data
from data_cleaner import clean_retail_data, validate_columns
from database import RetailDatabase
from eda import (
    plot_sales_trend, plot_category_revenue, plot_top_products,
    plot_store_performance, plot_payment_distribution,
    plot_demographics, plot_discount_vs_sales
)
from segmentation import (
    calculate_rfm, train_rfm_kmeans, plot_rfm_3d, plot_segment_distribution
)
from predictor import (
    prepare_time_series_dataset, train_and_evaluate_models,
    forecast_future_sales, plot_forecast_results
)
from insights import generate_business_insights

# Page Config
st.set_page_config(
    page_title="Retail Sales Analytics & Prediction System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .insight-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database connection
@st.cache_resource
def get_db_instance():
    return RetailDatabase()

db = get_db_instance()

# Data Loader Cache
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_uploader=None):
    if file_uploader is not None:
        try:
            raw_df = pd.read_csv(file_uploader)
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {e}")
            return None, None
    else:
        raw_path = os.path.join(os.path.dirname(__file__), "data", "raw_retail_sales.csv")
        if not os.path.exists(raw_path):
            raw_df = generate_retail_data(n_samples=3200, output_path=raw_path)
        else:
            raw_df = pd.read_csv(raw_path)
            
    try:
        cleaned_df, cleaning_summary = clean_retail_data(raw_df)
        # Store in SQLite database automatically
        db.save_dataframe(cleaned_df, "cleaned_transactions")
        return cleaned_df, cleaning_summary
    except Exception as e:
        st.error(f"Data Cleaning Error: {e}")
        return None, None

# Main Application
def main():
    st.markdown('<div class="main-header">🛍️ Retail Sales Analytics & Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End Business Intelligence, Customer RFM Segmentation & Sales Forecasting</div>', unsafe_allow_html=True)

    # Sidebar Data Controls
    st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=70)
    st.sidebar.header("📁 Data Source & Options")
    
    data_option = st.sidebar.radio(
        "Choose Data Input:",
        ["Sample Synthetic Data", "Upload CSV Dataset"]
    )
    
    uploaded_file = None
    if data_option == "Upload CSV Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload Retail CSV", type=["csv"])
        if uploaded_file is None:
            st.info("💡 Please upload a CSV file with retail transactions or switch to Sample Synthetic Data.")
            return

    # Load Data
    with st.spinner("Loading & Preprocessing Sales Dataset..."):
        df, summary = load_and_clean_data(uploaded_file)

    if df is None:
        return

    # Global Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Global Filters")
    
    # Date Range Filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
        
    # Store Location Filter
    all_stores = sorted(df['Store Location'].unique().tolist())
    selected_stores = st.sidebar.multiselect("Store Location", all_stores, default=all_stores)
    
    # Category Filter
    all_categories = sorted(df['Product Category'].unique().tolist())
    selected_categories = st.sidebar.multiselect("Product Category", all_categories, default=all_categories)
    
    # Payment Method Filter
    all_pm = sorted(df['Payment Method'].unique().tolist())
    selected_pm = st.sidebar.multiselect("Payment Method", all_pm, default=all_pm)

    # Filter Application
    filtered_df = df[
        (df['Date'].dt.date >= start_date) &
        (df['Date'].dt.date <= end_date) &
        (df['Store Location'].isin(selected_stores if selected_stores else all_stores)) &
        (df['Product Category'].isin(selected_categories if selected_categories else all_categories)) &
        (df['Payment Method'].isin(selected_pm if selected_pm else all_pm))
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Filtered Records: **{len(filtered_df):,}** / {len(df):,}")
    
    # Download Button for Cleaned Data
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=csv_bytes,
        file_name="cleaned_retail_sales_export.csv",
        mime="text/csv"
    )

    # Tabs Navigation
    tabs = st.tabs([
        "📊 KPI Overview",
        "📈 EDA & Trends",
        "🎯 Customer RFM",
        "🤖 Sales Forecast ML",
        "💡 Business Insights",
        "🗄️ SQLite Database"
    ])

    # TAB 1: KPI OVERVIEW
    with tabs[0]:
        st.subheader("📌 Business Executive KPI Dashboard")
        
        if len(filtered_df) == 0:
            st.warning("No data matching current filters!")
        else:
            total_rev = filtered_df['Total Amount'].sum()
            total_orders = filtered_df['Transaction ID'].nunique()
            total_custs = filtered_df['Customer ID'].nunique()
            aov = total_rev / total_orders if total_orders > 0 else 0
            total_units = filtered_df['Quantity'].sum()
            
            top_cat = filtered_df.groupby('Product Category')['Total Amount'].sum().idxmax()
            top_prod = filtered_df.groupby('Product Name')['Total Amount'].sum().idxmax()
            
            # Monthly Growth Rate
            monthly = filtered_df.groupby('Year_Month')['Total Amount'].sum()
            if len(monthly) >= 2:
                growth_rate = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
            else:
                growth_rate = 0.0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Revenue</div>
                    <div class="kpi-value">${total_rev:,.2f}</div>
                    <div class="kpi-subtext">MoM Growth: {growth_rate:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #10B981;">
                    <div class="kpi-title">Total Orders</div>
                    <div class="kpi-value">{total_orders:,}</div>
                    <div class="kpi-subtext">Units Sold: {total_units:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #F59E0B;">
                    <div class="kpi-title">Avg Order Value (AOV)</div>
                    <div class="kpi-value">${aov:,.2f}</div>
                    <div class="kpi-subtext">Per Transaction</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #8B5CF6;">
                    <div class="kpi-title">Active Customers</div>
                    <div class="kpi-value">{total_custs:,}</div>
                    <div class="kpi-subtext">Top Category: {top_cat}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns([2, 1])
            with col_l:
                freq_choice = st.selectbox("Sales Trend Frequency:", ["Monthly", "Daily", "Yearly"], index=0)
                freq_map = {"Monthly": "M", "Daily": "D", "Yearly": "Y"}
                fig_trend = plot_sales_trend(filtered_df, freq=freq_map[freq_choice])
                st.plotly_chart(fig_trend, use_container_width=True)
            with col_r:
                fig_store = plot_store_performance(filtered_df)
                st.plotly_chart(fig_store, use_container_width=True)

    # TAB 2: EDA & TRENDS
    with tabs[1]:
        st.subheader("📈 Exploratory Data Analysis & Deep Dives")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_category_revenue(filtered_df), use_container_width=True)
            st.plotly_chart(plot_payment_distribution(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_top_products(filtered_df, top_n=10), use_container_width=True)
            st.plotly_chart(plot_demographics(filtered_df), use_container_width=True)
            
        st.plotly_chart(plot_discount_vs_sales(filtered_df), use_container_width=True)

    # TAB 3: CUSTOMER RFM SEGMENTATION
    with tabs[2]:
        st.subheader("🎯 Customer Segmentation via RFM & K-Means Clustering")
        st.markdown("""
        RFM Analysis evaluates customers on **Recency** (days since last purchase), **Frequency** (total orders), and **Monetary Value** (total spent).
        K-Means clusters customers into 4 strategic tiers to customize retention & marketing strategy.
        """)
        
        rfm_raw = calculate_rfm(filtered_df if len(filtered_df) > 50 else df)
        rfm_seg, profile, _ = train_rfm_kmeans(rfm_raw, n_clusters=4)
        
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.plotly_chart(plot_rfm_3d(rfm_seg), use_container_width=True)
        with c_right:
            st.plotly_chart(plot_segment_distribution(rfm_seg), use_container_width=True)
            
        st.markdown("### 📋 Segmented Customers Table")
        selected_seg = st.selectbox(
            "Filter Table by Segment:",
            ["All Segments"] + list(rfm_seg['Customer_Segment'].unique())
        )
        
        display_rfm = rfm_seg if selected_seg == "All Segments" else rfm_seg[rfm_seg['Customer_Segment'] == selected_seg]
        st.dataframe(display_rfm[['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Customer_Segment']], use_container_width=True)
        
        # Download RFM CSV
        rfm_csv = rfm_seg.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Customer Segments CSV",
            rfm_csv,
            "rfm_customer_segmentation.csv",
            "text/csv"
        )

    # TAB 4: SALES FORECASTING ML
    with tabs[3]:
        st.subheader("🤖 Machine Learning Sales Prediction & Forecasting")
        st.markdown("Train Supervised ML models (Linear Regression, Random Forest, Gradient Boosting) on daily aggregated historical sales.")
        
        ts_df = prepare_time_series_dataset(filtered_df if len(filtered_df) > 100 else df)
        
        col_btn, col_slider = st.columns([1, 2])
        with col_btn:
            train_click = st.button("🚀 Train Sales Prediction Models", type="primary")
        with col_slider:
            forecast_horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=7)
            
        model_save_path = os.path.join(os.path.dirname(__file__), "models", "sales_forecaster.pkl")
        
        # Train or load cache
        with st.spinner("Training ML Models..."):
            eval_results = train_and_evaluate_models(ts_df, model_save_path=model_save_path)
            
        best_name = eval_results['best_model_name']
        st.success(f"🏆 Best Performing Model: **{best_name}**")
        
        # Metric Comparison Table
        st.markdown("#### 📊 Model Performance Comparison")
        st.dataframe(eval_results['metrics_df'], use_container_width=True)
        
        # Forecast Generation
        forecast_df = forecast_future_sales(
            ts_df,
            eval_results['best_model'],
            eval_results['feature_cols'],
            forecast_days=forecast_horizon
        )
        
        # Plot Chart
        fig_forecast = plot_forecast_results(eval_results, forecast_df)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast Data Table & Download
        with st.expander("📄 View & Export Predicted Sales Table"):
            st.dataframe(forecast_df, use_container_width=True)
            forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Sales Forecast CSV",
                forecast_csv,
                f"sales_forecast_{forecast_horizon}days.csv",
                "text/csv"
            )

    # TAB 5: BUSINESS INSIGHTS
    with tabs[4]:
        st.subheader("💡 Automated Business Insights & Recommendations")
        
        rfm_raw = calculate_rfm(filtered_df if len(filtered_df) > 50 else df)
        rfm_seg, _, _ = train_rfm_kmeans(rfm_raw, n_clusters=4)
        
        insights_res = generate_business_insights(filtered_df if len(filtered_df) > 0 else df, rfm_seg)
        
        col_ins, col_rec = st.columns(2)
        with col_ins:
            st.markdown("### 🔍 Key Findings")
            for item in insights_res['insights']:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>{item['icon']} {item['title']}</h4>
                    <p style="color: #475569;">{item['text']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_rec:
            st.markdown("### 🚀 Strategic Actionable Recommendations")
            for item in insights_res['recommendations']:
                st.markdown(f"""
                <div class="insight-card" style="border-left: 4px solid #10B981;">
                    <h4>{item['icon']} {item['title']} <span style="font-size:0.75rem; color:#10B981;">[{item['area']}]</span></h4>
                    <p style="color: #334155;">{item['action']}</p>
                </div>
                """, unsafe_allow_html=True)

    # TAB 6: SQLITE DATABASE ENGINE
    with tabs[5]:
        st.subheader("🗄️ SQLite Analytical Database Engine")
        st.markdown("Query the underlying SQLite database (`sales_analytics.db`) directly using standard SQL syntax.")
        
        default_sql = 'SELECT "Product Category", COUNT("Transaction ID") as total_orders, ROUND(SUM("Total Amount"), 2) as total_revenue FROM cleaned_transactions GROUP BY "Product Category" ORDER BY total_revenue DESC;'
        
        query_input = st.text_area("Write SQL Query:", value=default_sql, height=100)
        
        if st.button("▶️ Execute Query"):
            try:
                res_df = db.execute_query(query_input)
                st.success(f"Query executed successfully! Returned {len(res_df)} rows.")
                st.dataframe(res_df, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

if __name__ == "__main__":
    main()
