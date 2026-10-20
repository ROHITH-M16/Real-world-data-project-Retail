"""
Data Generator Module for Retail Sales Analytics System.
Generates realistic synthetic retail transactions dataset with missing values and outliers for testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

def generate_retail_data(n_samples=3000, random_seed=42, output_path=None):
    """
    Generate a realistic synthetic retail sales dataset.
    
    Args:
        n_samples (int): Number of transaction records to generate.
        random_seed (int): Seed for reproducibility.
        output_path (str): Optional path to save CSV file.
        
    Returns:
        pd.DataFrame: Generated retail sales dataset.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Categories & Products mapping with realistic base unit prices
    catalog = {
        "Electronics": [
            ("Wireless Headphones", 120.0),
            ("Smart Watch", 220.0),
            ("Bluetooth Speaker", 75.0),
            ("4K Monitor", 350.0),
            ("Ergonomic Mouse", 45.0),
            ("Mechanical Keyboard", 90.0)
        ],
        "Apparel": [
            ("Cotton T-Shirt", 25.0),
            ("Denim Jeans", 65.0),
            ("Hooded Sweatshirt", 50.0),
            ("Running Shoes", 110.0),
            ("Leather Jacket", 195.0),
            ("Winter Coat", 160.0)
        ],
        "Home & Kitchen": [
            ("Coffee Maker", 85.0),
            ("Air Fryer", 130.0),
            ("Blender", 60.0),
            ("Vacuum Cleaner", 210.0),
            ("Cookware Set", 175.0),
            ("Memory Foam Pillow", 40.0)
        ],
        "Beauty & Care": [
            ("Skincare Serum", 35.0),
            ("Electric Toothbrush", 70.0),
            ("Hair Dryer", 55.0),
            ("Perfume Gift Set", 95.0),
            ("Sunscreen SPF 50", 20.0)
        ],
        "Sports & Outdoors": [
            ("Yoga Mat", 30.0),
            ("Dumbbell Set", 80.0),
            ("Camping Tent", 150.0),
            ("Water Bottle", 18.0),
            ("Bicycle Helmet", 45.0)
        ]
    }
    
    # Product IDs catalog construction
    product_pool = []
    prod_id_counter = 501
    for cat, items in catalog.items():
        for name, price in items:
            product_pool.append({
                "Product_ID": f"PRD-{prod_id_counter}",
                "Category": cat,
                "Product_Name": name,
                "Base_Price": price
            })
            prod_id_counter += 1
            
    # Customers pool (400 unique customers for RFM clustering)
    customer_ids = [f"CUST-{i:03d}" for i in range(101, 501)]
    customer_gender = {cid: random.choice(["Male", "Female", "Non-Binary"]) for cid in customer_ids}
    customer_age = {cid: random.randint(18, 72) for cid in customer_ids}
    
    stores = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle", "Online Store"]
    payment_methods = ["Credit Card", "Debit Card", "UPI / Cash", "Net Banking", "E-Wallet"]
    
    # Dates spanning ~2 years
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 9, 30)
    total_days = (end_date - start_date).days
    
    rows = []
    for i in range(1, n_samples + 1):
        txn_id = f"TXN-{1000 + i}"
        days_offset = random.randint(0, total_days)
        txn_date = start_date + timedelta(days=days_offset)
        
        # Add seasonal bias (higher sales in Q4)
        if txn_date.month in [11, 12]:
            qty_multiplier = random.choices([1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.1])[0]
        else:
            qty_multiplier = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            
        cust_id = random.choice(customer_ids)
        product = random.choice(product_pool)
        
        qty = random.randint(1, 5) * qty_multiplier
        unit_price = round(product["Base_Price"] * random.uniform(0.95, 1.05), 2)
        discount_pct = random.choice([0, 5, 10, 15, 20, 25])
        
        total_amount = round(qty * unit_price * (1 - discount_pct / 100.0), 2)
        
        store = random.choice(stores)
        payment = random.choice(payment_methods)
        age = customer_age[cust_id]
        gender = customer_gender[cust_id]
        
        rows.append({
            "Transaction ID": txn_id,
            "Date": txn_date.strftime("%Y-%m-%d"),
            "Customer ID": cust_id,
            "Product ID": product["Product_ID"],
            "Product Category": product["Category"],
            "Product Name": product["Product_Name"],
            "Quantity": qty,
            "Unit Price": unit_price,
            "Total Amount": total_amount,
            "Discount (%)": discount_pct,
            "Payment Method": payment,
            "Store Location": store,
            "Customer Age": age,
            "Customer Gender": gender
        })
        
    df = pd.DataFrame(rows)
    
    # Introduce subtle real-world dirty data (duplicate rows, missing values, minor date formatting quirks)
    # 1. Duplicate 15 rows
    duplicates = df.sample(n=15, random_state=random_seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Add missing values (~1.5% in Customer Age, Payment Method, Store Location)
    missing_idx_age = np.random.choice(df.index, size=int(len(df) * 0.015), replace=False)
    df.loc[missing_idx_age, "Customer Age"] = np.nan
    
    missing_idx_pm = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[missing_idx_pm, "Payment Method"] = np.nan
    
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully saved to: {output_path}")
        
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_file = os.path.join(out_dir, "raw_retail_sales.csv")
    df = generate_retail_data(n_samples=3200, output_path=out_file)
    print(f"Generated {len(df)} transactions. Columns: {list(df.columns)}")
