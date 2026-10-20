"""
Customer Segmentation Module using RFM Analysis and K-Means Clustering.
Calculates Recency, Frequency, Monetary metrics, scales features, fits K-Means, and assigns business segments.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import pickle
import os

def calculate_rfm(df):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for each customer.
    """
    # Reference date is the day after the last transaction in the dataset
    max_date = pd.to_datetime(df['Date']).max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'Date': lambda x: (max_date - pd.to_datetime(x).max()).days, # Recency
        'Transaction ID': 'count',                                   # Frequency
        'Total Amount': 'sum'                                       # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    # Calculate RFM Quartile Scores (1 to 4)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop') # Lower recency is better
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    return rfm

def train_rfm_kmeans(rfm_df, n_clusters=4, model_save_path=None):
    """
    Scales RFM features and trains K-Means model.
    Assigns human-readable segment names based on cluster metrics.
    """
    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Log transform to reduce skewness
    features_log = np.log1p(features)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Profile clusters by average RFM values to map to business segment names dynamically
    cluster_profile = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # Rank clusters based on a custom score (high monetary & frequency, low recency)
    # Scaled metric ranking
    cluster_profile['Rank_Score'] = (
        (1.0 / (cluster_profile['Recency'] + 1)) * 0.3 + 
        cluster_profile['Frequency'] * 0.3 + 
        cluster_profile['Monetary'] * 0.4
    )
    
    cluster_profile = cluster_profile.sort_values(by='Rank_Score', ascending=False).reset_index(drop=True)
    
    # Define Segment Names order
    segment_names = [
        "High Value Customers",
        "Regular / Loyal Customers",
        "New / Potential Customers",
        "At-Risk / Inactive Customers"
    ]
    
    cluster_mapping = {}
    for i, row in cluster_profile.iterrows():
        cluster_id = int(row['Cluster'])
        cluster_mapping[cluster_id] = segment_names[min(i, len(segment_names) - 1)]
        
    rfm_df['Customer_Segment'] = rfm_df['Cluster'].map(cluster_mapping)
    
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'kmeans': kmeans, 'mapping': cluster_mapping}, f)
        print(f"Saved RFM KMeans model to: {model_save_path}")
        
    return rfm_df, cluster_profile, kmeans

def plot_rfm_3d(rfm_df):
    """
    Generate 3D scatter plot of RFM Clusters.
    """
    fig = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Customer_Segment',
        hover_data=['Customer_ID'],
        title="3D Customer Segmentation (RFM Analysis)",
        labels={
            'Recency': 'Recency (Days)',
            'Frequency': 'Frequency (Orders)',
            'Monetary': 'Monetary Value ($)'
        },
        color_discrete_map={
            "High Value Customers": "#10B981",
            "Regular / Loyal Customers": "#2563EB",
            "New / Potential Customers": "#F59E0B",
            "At-Risk / Inactive Customers": "#EF4444"
        },
        template="plotly_white"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_segment_distribution(rfm_df):
    """
    Bar chart showing customer count and total revenue share per segment.
    """
    seg_summary = rfm_df.groupby('Customer_Segment').agg(
        Customer_Count=('Customer_ID', 'count'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Monetary=('Monetary', 'mean')
    ).reset_index()

    fig = px.bar(
        seg_summary,
        x='Customer_Segment',
        y='Customer_Count',
        color='Customer_Segment',
        title="Customer Distribution by Segment",
        labels={'Customer_Count': 'Number of Customers', 'Customer_Segment': 'Segment'},
        text_auto=True,
        color_discrete_map={
            "High Value Customers": "#10B981",
            "Regular / Loyal Customers": "#2563EB",
            "New / Potential Customers": "#F59E0B",
            "At-Risk / Inactive Customers": "#EF4444"
        },
        template="plotly_white"
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig

if __name__ == "__main__":
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "rfm_kmeans.pkl")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        rfm_df = calculate_rfm(df)
        rfm_segmented, profile, model = train_rfm_kmeans(rfm_df, n_clusters=4, model_save_path=model_path)
        print("RFM Customer Segmentation Summary:")
        print(rfm_segmented['Customer_Segment'].value_counts())
