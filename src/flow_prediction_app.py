import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def nash_sutcliffe_efficiency(observed, predicted):
    """Calculate Nash-Sutcliffe Efficiency"""
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    return 1 - (numerator / denominator)

def calculate_metrics(observed, predicted):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    mae = mean_absolute_error(observed, predicted)
    correlation = np.corrcoef(observed, predicted)[0, 1]
    bias = np.mean(predicted - observed)
    nse = nash_sutcliffe_efficiency(observed, predicted)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'BIAS': bias,
        'Nash_Sutcliffe': nse
    }

def process_flow_data():
    """Load and process flow data to monthly scale, filtering negative values"""
    print("Processing flow data...")
    
    # Load flow data
    flow_data = pd.read_excel('../data/Vazao_FUNIL.xlsx', sheet_name='Vazao_FUNIL')
    
    # Convert Data column to datetime
    flow_data['Data'] = pd.to_datetime(flow_data['Data'])
    
    # Filter out negative values (replace with NaN)
    target_cols = [58030000, 58060000]
    for col in target_cols:
        flow_data.loc[flow_data[col] < 0, col] = np.nan
    
    # Convert to monthly data
    flow_data.set_index('Data', inplace=True)
    monthly_flow = flow_data[target_cols].resample('M').mean()
    
    # Create year and month columns
    monthly_flow['year'] = monthly_flow.index.year
    monthly_flow['month'] = monthly_flow.index.month
    
    # Reset index
    monthly_flow.reset_index(inplace=True)
    
    print(f"Monthly flow data shape: {monthly_flow.shape}")
    print(f"Date range: {monthly_flow['Data'].min()} to {monthly_flow['Data'].max()}")
    
    return monthly_flow

def load_meteorological_data():
    """Load meteorological data"""
    print("Loading meteorological data...")
    
    met_data = pd.read_csv('../data/glob-funil-subbasin.csv')
    
    # Select relevant columns
    predictor_cols = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
    time_cols = ['year_x', 'month_x']
    id_col = ['ID_Subbasin']
    
    met_data_clean = met_data[predictor_cols + time_cols + id_col].copy()
    
    # Remove rows with NaN in ID_Subbasin
    met_data_clean = met_data_clean.dropna(subset=['ID_Subbasin'])
    
    print(f"Meteorological data shape: {met_data_clean.shape}")
    
    return met_data_clean

def merge_data(monthly_flow, met_data):
    """Merge meteorological and flow data"""
    print("Merging datasets...")
    
    # Mapping: flow column -> subbasin ID
    flow_to_subbasin = {
        58030000: 24,
        58060000: 36
    }
    
    merged_datasets = {}
    
    for flow_col, subbasin_id in flow_to_subbasin.items():
        # Filter meteorological data for this subbasin
        met_subset = met_data[met_data['ID_Subbasin'] == subbasin_id].copy()
        
        # Merge on year and month
        merged = pd.merge(
            monthly_flow[['year', 'month', flow_col]],
            met_subset,
            left_on=['year', 'month'],
            right_on=['year_x', 'month_x'],
            how='inner'
        )
        
        # Remove rows with NaN in flow data
        merged = merged.dropna(subset=[flow_col])
        
        merged_datasets[flow_col] = merged
        
        print(f"Merged data for {flow_col} (subbasin {subbasin_id}): {merged.shape}")
    
    return merged_datasets

def train_models(data, target_col, train_year_cutoff=2015):
    """Train multiple ML models"""
    predictor_cols = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
    
    # Split data
    train_data = data[data['year_x'] <= train_year_cutoff]
    test_data = data[data['year_x'] > train_year_cutoff]
    
    if len(train_data) == 0 or len(test_data) == 0:
        print(f"Warning: Insufficient data for {target_col}")
        return None
    
    X_train = train_data[predictor_cols]
    y_train = train_data[target_col]
    X_test = test_data[predictor_cols]
    y_test = test_data[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    results = {}
    predictions = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name} for {target_col}...")
        
        # Train model
        if model_name in ['Linear_Regression', 'Ridge', 'SVR']:
            model.fit(X_train_scaled, y_train)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        predictions[model_name] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_train': y_train,
            'y_test': y_test,
            'train_dates': train_data[['year_x', 'month_x']],
            'test_dates': test_data[['year_x', 'month_x']]
        }
    
    return results, predictions

def main():
    """Main function"""
    print("Starting Flow Prediction Application")
    print("=" * 50)
    
    # Process data
    monthly_flow = process_flow_data()
    met_data = load_meteorological_data()
    merged_datasets = merge_data(monthly_flow, met_data)
    
    # Train models for each flow station
    all_results = {}
    all_predictions = {}
    
    for flow_col in [58030000, 58060000]:
        print(f"\n{'='*20} Processing {flow_col} {'='*20}")
        
        if flow_col in merged_datasets:
            results, predictions = train_models(merged_datasets[flow_col], flow_col)
            if results:
                all_results[flow_col] = results
                all_predictions[flow_col] = predictions
    
    return all_results, all_predictions, merged_datasets

if __name__ == "__main__":
    results, predictions, datasets = main()