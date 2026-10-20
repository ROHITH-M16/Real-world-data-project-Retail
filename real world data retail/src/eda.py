"""
Exploratory Data Analysis (EDA) Module.
Provides interactive Plotly visualization functions for sales trends, categories, products, stores, demographics, and discounts.
"""

import plotly.express as px
import plotly.graph_objects as px_go
import pandas as pd
import numpy as np

# Theme palette
COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#10B981"
COLOR_ACCENT = "#F59E0B"
COLOR_DANGER = "#EF4444"
COLOR_PURPLE = "#8B5CF6"

TEMPLATE = "plotly_white"

def plot_sales_trend(df, freq='M'):
    """
    Plots sales trends over time (Daily, Monthly, or Yearly).
    """
    df_copy = df.copy()
    if freq == 'D':
        trend = df_copy.groupby('Date')['Total Amount'].sum().reset_index()
        title = "Daily Sales Revenue Trend"
        x_col = 'Date'
    elif freq == 'Y':
        trend = df_copy.groupby('Year')['Total Amount'].sum().reset_index()
        trend['Year'] = trend['Year'].astype(str)
        title = "Yearly Sales Revenue Trend"
        x_col = 'Year'
    else: # Default Monthly
        trend = df_copy.groupby('Year_Month')['Total Amount'].sum().reset_index()
        title = "Monthly Sales Revenue Trend"
        x_col = 'Year_Month'

    fig = px.line(
        trend, 
        x=x_col, 
        y='Total Amount',
        title=title,
        labels={x_col: 'Time Period', 'Total Amount': 'Revenue ($)'},
        markers=True,
        template=TEMPLATE
    )
    fig.update_traces(line_color=COLOR_PRIMARY, line_width=3, marker=dict(size=7))
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_category_revenue(df):
    """
    Bar chart showing total revenue and average discount by product category.
    """
    cat_df = df.groupby('Product Category').agg(
        Total_Revenue=('Total Amount', 'sum'),
        Total_Units=('Quantity', 'sum'),
        Avg_Discount=('Discount (%)', 'mean')
    ).reset_index().sort_values(by='Total_Revenue', ascending=True)

    fig = px.bar(
        cat_df,
        x='Total_Revenue',
        y='Product Category',
        orientation='h',
        title="Revenue by Product Category",
        labels={'Total_Revenue': 'Revenue ($)', 'Product Category': 'Category'},
        text_auto='.2s',
        color='Total_Revenue',
        color_continuous_scale='Viridis',
        template=TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
    return fig

def plot_top_products(df, top_n=10):
    """
    Horizontal bar chart of top N selling products by revenue.
    """
    prod_df = df.groupby(['Product Name', 'Product Category']).agg(
        Total_Revenue=('Total Amount', 'sum'),
        Total_Quantity=('Quantity', 'sum')
    ).reset_index().sort_values(by='Total_Revenue', ascending=False).head(top_n)

    fig = px.bar(
        prod_df.sort_values(by='Total_Revenue', ascending=True),
        x='Total_Revenue',
        y='Product Name',
        color='Product Category',
        orientation='h',
        title=f"Top {top_n} Selling Products by Revenue",
        labels={'Total_Revenue': 'Total Revenue ($)', 'Product Name': 'Product'},
        text_auto='.2s',
        template=TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_store_performance(df):
    """
    Bar chart comparing store locations by revenue and average order size.
    """
    store_df = df.groupby('Store Location').agg(
        Total_Revenue=('Total Amount', 'sum'),
        Total_Orders=('Transaction ID', 'count'),
        Avg_Order_Value=('Total Amount', 'mean')
    ).reset_index().sort_values(by='Total_Revenue', ascending=False)

    fig = px.bar(
        store_df,
        x='Store Location',
        y='Total_Revenue',
        color='Avg_Order_Value',
        title="Sales Revenue by Store Location",
        labels={'Total_Revenue': 'Revenue ($)', 'Avg_Order_Value': 'Avg Order ($)'},
        text_auto='.2s',
        color_continuous_scale='Tealgrn',
        template=TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_payment_distribution(df):
    """
    Donut chart showing breakdown of payment methods.
    """
    pm_df = df.groupby('Payment Method')['Total Amount'].sum().reset_index()
    fig = px.pie(
        pm_df,
        values='Total Amount',
        names='Payment Method',
        title="Payment Method Breakdown",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template=TEMPLATE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_demographics(df):
    """
    Grouped bar chart showing revenue by Age Group and Gender.
    """
    demo_df = df.groupby(['Age_Group', 'Customer Gender'])['Total Amount'].sum().reset_index()
    fig = px.bar(
        demo_df,
        x='Age_Group',
        y='Total Amount',
        color='Customer Gender',
        barmode='group',
        title="Revenue by Customer Demographics (Age Group & Gender)",
        labels={'Total Amount': 'Total Revenue ($)', 'Age_Group': 'Age Group'},
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT],
        template=TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_discount_vs_sales(df):
    """
    Scatter plot showing relationship between Discount (%) and Total Amount / Quantity.
    """
    fig = px.scatter(
        df,
        x='Discount (%)',
        y='Total Amount',
        color='Product Category',
        size='Quantity',
        hover_data=['Product Name', 'Store Location'],
        title="Impact of Discount (%) on Sales Volume & Revenue",
        labels={'Discount (%)': 'Discount Percentage (%)', 'Total Amount': 'Transaction Total ($)'},
        opacity=0.7,
        template=TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

if __name__ == "__main__":
    import os
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        fig = plot_sales_trend(df)
        print("Successfully generated sales trend plot object!")
