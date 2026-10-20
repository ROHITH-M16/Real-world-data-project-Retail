"""
SQLite Database Operations Module.
Manages storing raw and cleaned sales data into SQLite database, and running aggregate SQL queries.
"""

import sqlite3
import pandas as pd
import os
import logging

class RetailDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(__file__), "..", "database")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "sales_analytics.db")
        self.db_path = db_path

    def get_connection(self):
        """Establish and return SQLite connection."""
        return sqlite3.connect(self.db_path)

    def save_dataframe(self, df, table_name, if_exists="replace"):
        """Save a pandas DataFrame into SQLite table."""
        with self.get_connection() as conn:
            # Convert timestamp columns to string for SQLite compatibility
            df_to_save = df.copy()
            for col in df_to_save.select_dtypes(include=['datetime64[ns]']).columns:
                df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d')
            df_to_save.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logging.info(f"Saved {len(df_to_save)} rows to table '{table_name}' in SQLite DB.")

    def load_dataframe(self, table_name):
        """Load data from SQLite table into pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df

    def execute_query(self, query):
        """Run custom SQL query and return result DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_kpi_summary_sql(self):
        """Run analytical SQL query to get high-level business KPIs."""
        query = """
        SELECT 
            COUNT(DISTINCT "Transaction ID") AS total_orders,
            COUNT(DISTINCT "Customer ID") AS total_customers,
            ROUND(SUM("Total Amount"), 2) AS total_revenue,
            ROUND(AVG("Total Amount"), 2) AS avg_order_value,
            SUM("Quantity") AS total_units_sold
        FROM cleaned_transactions;
        """
        try:
            return self.execute_query(query)
        except Exception as e:
            logging.error(f"Error fetching KPI summary from SQL: {e}")
            return None

if __name__ == "__main__":
    db = RetailDatabase()
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        db.save_dataframe(df, "cleaned_transactions")
        kpis = db.get_kpi_summary_sql()
        print("SQL KPI Summary:")
        print(kpis)
