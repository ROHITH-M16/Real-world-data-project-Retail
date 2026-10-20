"""
Data Cleaner and Preprocessor Module.
Handles validation, cleaning, missing value imputation, outlier detection, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUIRED_COLUMNS = [
    'Transaction ID', 'Date', 'Customer ID', 'Product ID',
    'Product Category', 'Product Name', 'Quantity', 'Unit Price',
    'Total Amount', 'Discount (%)', 'Payment Method', 'Store Location',
    'Customer Age', 'Customer Gender'
]

COLUMN_MAPPING_VARIANTS = {
    'Transaction_ID': 'Transaction ID',
    'Customer_ID': 'Customer ID',
    'Product_ID': 'Product ID',
    'Product_Category': 'Product Category',
    'Product_Name': 'Product Name',
    'Unit_Price': 'Unit Price',
    'Total_Amount': 'Total Amount',
    'Discount': 'Discount (%)',
    'Discount_Pct': 'Discount (%)',
    'Payment_Method': 'Payment Method',
    'Store_Location': 'Store Location',
    'Customer_Age': 'Customer Age',
    'Customer_Gender': 'Customer Gender'
}

def validate_columns(df):
    """
    Validate that the DataFrame contains required retail sales columns.
    Maps common column name variants if needed.
    """
    df = df.copy()
    # Rename matching variants
    df = df.rename(columns=COLUMN_MAPPING_VARIANTS)
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}", df
    return True, "All required columns are present.", df

def clean_retail_data(df):
    """
    Perform full data cleaning and feature engineering.
    
    Returns:
        tuple: (cleaned_df, cleaning_summary_dict)
    """
    is_valid, msg, df = validate_columns(df)
    if not is_valid:
        raise ValueError(msg)
        
    initial_count = len(df)
    cleaning_summary = {
        'initial_rows': initial_count,
        'duplicates_removed': 0,
        'missing_imputed': {},
        'outliers_clipped': 0,
        'final_rows': 0
    }
    
    # 1. Remove duplicate transactions
    df = df.drop_duplicates(subset=['Transaction ID']).reset_index(drop=True)
    cleaning_summary['duplicates_removed'] = initial_count - len(df)
    
    # 2. Date conversion
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Drop rows with unparseable dates if any
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    
    # 3. Numeric Conversions & Invalid Value Handling
    numeric_cols = ['Quantity', 'Unit Price', 'Total Amount', 'Discount (%)', 'Customer Age']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Enforce non-negative values for quantities and prices
    df['Quantity'] = df['Quantity'].apply(lambda x: max(1, x) if pd.notnull(x) else x)
    df['Unit Price'] = df['Unit Price'].apply(lambda x: max(0.01, x) if pd.notnull(x) else x)
    df['Discount (%)'] = df['Discount (%)'].apply(lambda x: min(100.0, max(0.0, x)) if pd.notnull(x) else 0.0)
    
    # Recalculate Total Amount if invalid or missing
    df['Total Amount'] = np.where(
        df['Total Amount'].isnull() | (df['Total Amount'] <= 0),
        df['Quantity'] * df['Unit Price'] * (1 - df['Discount (%)'] / 100.0),
        df['Total Amount']
    )
    
    # 4. Impute Missing Values
    # Age -> median
    if df['Customer Age'].isnull().sum() > 0:
        median_age = df['Customer Age'].median()
        df['Customer Age'] = df['Customer Age'].fillna(round(median_age))
        cleaning_summary['missing_imputed']['Customer Age'] = f"Imputed with median ({round(median_age)})"
        
    # Categoricals -> mode or 'Unknown'
    cat_cols = ['Payment Method', 'Store Location', 'Customer Gender', 'Product Category']
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            cleaning_summary['missing_imputed'][col] = f"Imputed with mode ({mode_val})"
            
    # 5. Outlier Detection & Treatment for Total Amount (IQR clipping to 99th percentile)
    q99 = df['Total Amount'].quantile(0.99)
    outliers_mask = df['Total Amount'] > q99 * 2.5 # extreme spikes
    cleaning_summary['outliers_clipped'] = int(outliers_mask.sum())
    df['Total Amount'] = np.where(df['Total Amount'] > q99 * 2.5, q99 * 2.5, df['Total Amount'])
    
    # 6. Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Day'] = df['Date'].dt.day
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    
    df['Gross_Amount'] = round(df['Quantity'] * df['Unit Price'], 2)
    df['Discount_Amount'] = round(df['Gross_Amount'] - df['Total Amount'], 2)
    
    # Customer Age Bins
    age_bins = [0, 25, 35, 50, 65, 120]
    age_labels = ['18-25', '26-35', '36-50', '51-65', '65+']
    df['Age_Group'] = pd.cut(df['Customer Age'], bins=age_bins, labels=age_labels, right=True)
    df['Age_Group'] = df['Age_Group'].astype(str).fillna('Unknown')
    
    # Sort chronologically by Date
    df = df.sort_values(by='Date').reset_index(drop=True)
    cleaning_summary['final_rows'] = len(df)
    
    return df, cleaning_summary

if __name__ == "__main__":
    import os
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_retail_sales.csv")
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        cleaned_df, summary = clean_retail_data(raw_df)
        print("Data Cleaning Summary:", summary)
        print("Cleaned shape:", cleaned_df.shape)
        cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"Cleaned dataset saved to: {cleaned_path}")
