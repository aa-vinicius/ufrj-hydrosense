import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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

from metricas_gutemberg import nse, kge, rmse, mae, r2, pbias, bias

def calculate_metrics(observed, predicted):
    """Calculate performance metrics (todas as métricas hidrológicas)"""
    metrics = {}
    try:
        metrics['RMSE'] = rmse(observed, predicted)
    except Exception:
        metrics['RMSE'] = np.nan
    try:
        metrics['MAE'] = mae(observed, predicted)
    except Exception:
        metrics['MAE'] = np.nan
    try:
        metrics['R2'] = r2(observed, predicted)
    except Exception:
        metrics['R2'] = np.nan
    try:
        metrics['Correlation'] = np.corrcoef(observed, predicted)[0,1] if len(observed) > 1 else np.nan
    except Exception:
        metrics['Correlation'] = np.nan
    try:
        metrics['Nash_Sutcliffe'] = nse(observed, predicted)
    except Exception:
        metrics['Nash_Sutcliffe'] = np.nan
    try:
        metrics['KGE'] = kge(observed, predicted)
    except Exception:
        metrics['KGE'] = np.nan
    try:
        metrics['PBIAS'] = pbias(observed, predicted)
    except Exception:
        metrics['PBIAS'] = np.nan
    try:
        metrics['Bias'] = bias(observed, predicted)
    except Exception:
        metrics['Bias'] = np.nan
    return metrics

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
    monthly_flow = flow_data[target_cols].resample('ME').mean()
    
    # Create year and month columns
    monthly_flow['year'] = monthly_flow.index.year
    monthly_flow['month'] = monthly_flow.index.month
    
    # Reset index
    monthly_flow.reset_index(inplace=True)
    
    print(f"Monthly flow data shape: {monthly_flow.shape}")
    print(f"Date range: {monthly_flow['Data'].min()} to {monthly_flow['Data'].max()}")
    
    return monthly_flow

def load_meteorological_data():
    """Carrega os dados meteorológicos já no novo formato"""
    print("Loading meteorological data...")
    met_data = pd.read_csv('data/meteo_vazao_shifted_station_58030000.csv')
    # Não há mais sufixos _x/_y, e as colunas já estão padronizadas
    predictor_cols = ['year', 'month', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
    id_col = ['subbasin_id', 'station_id']
    target_col = ['flow_next_month']
    met_data_clean = met_data[predictor_cols + id_col + target_col].copy()
    print(f"Meteorological data shape: {met_data_clean.shape}")
    return met_data_clean

def merge_data(flow_data, met_data):
    """Com a nova estrutura, não é necessário merge externo. Apenas filtra por estação/subbacia se necessário."""
    if flow_data is None or met_data is None or flow_data.empty or met_data.empty:
        return {}
    print("Preparando datasets por estação...")
    merged_datasets = {}
    for station_id in met_data['station_id'].unique():
        merged = met_data[met_data['station_id'] == station_id].copy()
        merged_datasets[station_id] = merged
        print(f"Dataset para estação {station_id}: {merged.shape}")
    return merged_datasets

def train_models(data, target_col='flow_next_month', train_year_cutoff=2019):
    """Treina múltiplos modelos de ML com a nova estrutura de dados"""
    predictor_cols = ['year', 'month', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
    if data is None or data.empty or not all(col in data.columns for col in predictor_cols + [target_col]):
        print(f"Warning: DataFrame vazio ou colunas ausentes para {target_col}")
        return None, None

    # Remove linhas com NaN nas colunas preditoras ou alvo
    data_clean = data.dropna(subset=predictor_cols + [target_col])
    if data_clean.empty:
        print(f"Warning: Todos os dados possuem NaN para {target_col}")
        return None, None

    # Split data
    train_data = data_clean[(data_clean['year'] >= 1998) & (data_clean['year'] <= train_year_cutoff)]
    test_data = data_clean[(data_clean['year'] >= 2020) & (data_clean['year'] <= 2024)]
    if len(train_data) == 0 or len(test_data) == 0:
        print(f"Warning: Insufficient data for {target_col}")
        return None, None

    X_train = train_data[predictor_cols]
    y_train = train_data[target_col]
    X_test = test_data[predictor_cols]
    y_test = test_data[target_col]

    # Garante que não há NaN após o split
    if X_train.isnull().any().any() or y_train.isnull().any() or X_test.isnull().any().any() or y_test.isnull().any():
        print(f"Warning: Dados de treino/teste ainda possuem NaN para {target_col}")
        return None, None

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
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    results = {}
    predictions = {}
    for model_name, model in models.items():
        print(f"Training {model_name} for {target_col}...")
        if model_name in ['Linear_Regression', 'Ridge', 'SVR', 'MLP']:
            model.fit(X_train_scaled, y_train)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
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
            'train_dates': train_data[['year', 'month']],
            'test_dates': test_data[['year', 'month']]
        }
    return results, predictions

def main():
    """Main function"""
    print("Starting Flow Prediction Application")
    print("=" * 50)
    # Carrega dados já no novo formato
    met_data = load_meteorological_data()
    merged_datasets = merge_data(met_data, met_data)
    # Treina modelos para cada estação
    all_results = {}
    all_predictions = {}
    for station_id in merged_datasets:
        print(f"\n{'='*20} Processing station {station_id} {'='*20}")
        results, predictions = train_models(merged_datasets[station_id], 'flow_next_month')
        if results:
            all_results[station_id] = results
            all_predictions[station_id] = predictions
    return all_results, all_predictions, merged_datasets

if __name__ == "__main__":
    results, predictions, datasets = main()