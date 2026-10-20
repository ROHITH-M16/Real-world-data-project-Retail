"""
Sales Prediction and Forecasting Module.
Prepares daily time-series features, trains ML models (Linear Regression, Random Forest, Gradient Boosting),
evaluates metrics (MAE, MSE, RMSE, R²), saves best model, and generates future N-day predictions.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

def prepare_time_series_dataset(df, freq='D'):
    """
    Aggregates transaction total revenue to daily frequency and generates lag/rolling features.
    """
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Resample daily revenue
    daily_sales = df_copy.groupby(pd.Grouper(key='Date', freq=freq))['Total Amount'].sum().reset_index()
    daily_sales = daily_sales.rename(columns={'Total Amount': 'Revenue'}).sort_values(by='Date').reset_index(drop=True)
    
    # Fill any missing dates with 0 revenue
    full_date_range = pd.date_range(start=daily_sales['Date'].min(), end=daily_sales['Date'].max(), freq=freq)
    daily_sales = daily_sales.set_index('Date').reindex(full_date_range).fillna(0).reset_index()
    daily_sales = daily_sales.rename(columns={'index': 'Date'})
    
    # Time & Calendar Features
    daily_sales['Day_of_Week'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Month'] = daily_sales['Date'].dt.month
    daily_sales['Day_of_Month'] = daily_sales['Date'].dt.day
    daily_sales['Is_Weekend'] = daily_sales['Day_of_Week'].isin([5, 6]).astype(int)
    
    # Lag Features
    for lag in [1, 7, 14, 30]:
        daily_sales[f'Lag_{lag}'] = daily_sales['Revenue'].shift(lag)
        
    # Rolling Window Features
    daily_sales['Rolling_Mean_7'] = daily_sales['Revenue'].shift(1).rolling(window=7).mean()
    daily_sales['Rolling_Mean_14'] = daily_sales['Revenue'].shift(1).rolling(window=14).mean()
    daily_sales['Rolling_Std_7'] = daily_sales['Revenue'].shift(1).rolling(window=7).std()
    
    # Drop rows with NaN from lags
    cleaned_ts = daily_sales.dropna().reset_index(drop=True)
    return cleaned_ts

def train_and_evaluate_models(ts_df, test_ratio=0.2, model_save_path=None):
    """
    Splits data chronologically, trains LR, Random Forest, and Gradient Boosting models,
    compares MAE, MSE, RMSE, R², and saves best model.
    """
    feature_cols = [c for c in ts_df.columns if c not in ['Date', 'Revenue']]
    X = ts_df[feature_cols]
    y = ts_df['Revenue']
    
    # Chronological Train-Test Split
    split_idx = int(len(ts_df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = ts_df['Date'].iloc[split_idx:]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    metrics_list = []
    predictions_dict = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Enforce non-negative sales predictions
        y_pred = np.clip(y_pred, 0, None)
        predictions_dict[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics_list.append({
            'Model': name,
            'MAE ($)': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE ($)': round(rmse, 2),
            'R² Score': round(r2, 4)
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model
            
    metrics_df = pd.DataFrame(metrics_list).sort_values(by='R² Score', ascending=False)
    
    if model_save_path and best_model_obj:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': best_model_obj,
                'model_name': best_model_name,
                'feature_cols': feature_cols
            }, f)
        print(f"Saved best model ({best_model_name}) to: {model_save_path}")
        
    eval_results = {
        'metrics_df': metrics_df,
        'predictions_dict': predictions_dict,
        'test_dates': test_dates,
        'y_test': y_test,
        'best_model_name': best_model_name,
        'best_model': best_model_obj,
        'feature_cols': feature_cols,
        'last_ts_row': ts_df.iloc[-1]
    }
    
    return eval_results

def forecast_future_sales(ts_df, best_model, feature_cols, forecast_days=30):
    """
    Generates N-day recursive sales predictions into the future.
    """
    ts_working = ts_df.copy()
    last_date = ts_working['Date'].max()
    
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_preds = []
    
    for next_date in future_dates:
        # Build features for next_date based on current ts_working history
        recent_rev = ts_working['Revenue'].values
        
        day_of_week = next_date.dayofweek
        month = next_date.month
        day_of_month = next_date.day
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        lag_1 = recent_rev[-1]
        lag_7 = recent_rev[-7] if len(recent_rev) >= 7 else recent_rev[0]
        lag_14 = recent_rev[-14] if len(recent_rev) >= 14 else recent_rev[0]
        lag_30 = recent_rev[-30] if len(recent_rev) >= 30 else recent_rev[0]
        
        rolling_7 = np.mean(recent_rev[-7:])
        rolling_14 = np.mean(recent_rev[-14:])
        rolling_std_7 = np.std(recent_rev[-7:]) if len(recent_rev) >= 7 else 0.0
        
        feat_dict = {
            'Day_of_Week': day_of_week,
            'Month': month,
            'Day_of_Month': day_of_month,
            'Is_Weekend': is_weekend,
            'Lag_1': lag_1,
            'Lag_7': lag_7,
            'Lag_14': lag_14,
            'Lag_30': lag_30,
            'Rolling_Mean_7': rolling_7,
            'Rolling_Mean_14': rolling_14,
            'Rolling_Std_7': rolling_std_7
        }
        
        feat_row = pd.DataFrame([feat_dict])[feature_cols]
        pred_val = float(best_model.predict(feat_row)[0])
        pred_val = max(0.0, pred_val) # non-negative
        
        future_preds.append(pred_val)
        
        # Append to ts_working so subsequent lags can use it
        new_row = {
            'Date': next_date,
            'Revenue': pred_val,
            'Day_of_Week': day_of_week,
            'Month': month,
            'Day_of_Month': day_of_month,
            'Is_Weekend': is_weekend,
            'Lag_1': lag_1,
            'Lag_7': lag_7,
            'Lag_14': lag_14,
            'Lag_30': lag_30,
            'Rolling_Mean_7': rolling_7,
            'Rolling_Mean_14': rolling_14,
            'Rolling_Std_7': rolling_std_7
        }
        ts_working = pd.concat([ts_working, pd.DataFrame([new_row])], ignore_index=True)
        
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Revenue': [round(p, 2) for p in future_preds]
    })
    return forecast_df

def plot_forecast_results(eval_results, forecast_df=None):
    """
    Plots Actual vs Predicted sales on Test Set, and appends future forecast.
    """
    test_dates = eval_results['test_dates']
    y_test = eval_results['y_test']
    best_name = eval_results['best_model_name']
    best_preds = eval_results['predictions_dict'][best_name]
    
    fig = go.Figure()
    
    # Actual Sales line
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines+markers',
        name='Actual Sales ($)',
        line=dict(color='#2563EB', width=2),
        marker=dict(size=5)
    ))
    
    # Model Test Predictions line
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=best_preds,
        mode='lines',
        name=f'Predicted ({best_name})',
        line=dict(color='#10B981', width=2, dash='dash')
    ))
    
    # Future Forecast line if provided
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Revenue'],
            mode='lines+markers',
            name='30-Day Future Forecast ($)',
            line=dict(color='#F59E0B', width=3),
            marker=dict(size=6)
        ))
        
    fig.update_layout(
        title=f"Sales Forecast & Model Evaluation ({best_name})",
        xaxis_title="Date",
        yaxis_title="Daily Revenue ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

if __name__ == "__main__":
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail_sales.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "sales_forecaster.pkl")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        ts_df = prepare_time_series_dataset(df)
        eval_res = train_and_evaluate_models(ts_df, model_save_path=model_path)
        print("Model Performance Comparison:")
        print(eval_res['metrics_df'])
        
        forecast_df = forecast_future_sales(
            ts_df, 
            eval_res['best_model'], 
            eval_res['feature_cols'], 
            forecast_days=30
        )
        print("\nNext 5 Days Sales Forecast:")
        print(forecast_df.head())
