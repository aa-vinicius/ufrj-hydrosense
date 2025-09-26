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
import os
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

def load_meteorological_data():
    """
    Carrega e processa múltiplos arquivos de dados meteorológicos da pasta 'data'.
    Agrega os dados por mês, calculando a média para registros duplicados.
    """
    print("Loading and processing meteorological data from all station files...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    
    all_station_data = []
    
    # Itera sobre todos os arquivos CSV no diretório de dados
    for filename in os.listdir(data_dir):
        if filename.startswith('meteo_vazao_shifted_station_') and filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing file: {filename}")
            
            try:
                # Detecta o separador automaticamente
                station_data = pd.read_csv(file_path, sep=None, engine='python')
                
                # Colunas a serem mantidas e agrupadas
                predictor_cols = ['year', 'month', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
                id_cols = ['subbasin_id', 'station_id']
                target_col = ['flow_next_month']
                
                # Garante que todas as colunas necessárias existem
                required_cols = predictor_cols + id_cols + target_col
                if not all(col in station_data.columns for col in required_cols):
                    print(f"  [Warning] Skipping file {filename} due to missing columns.")
                    continue


                # Agrupa por ano e mês, calculando a média das outras colunas
                # subbasin_id não faz mais sentido após o agrupamento, então não será mantido
                agg_dict = {col: 'mean' for col in predictor_cols + target_col if col not in ['year', 'month']}
                agg_dict['station_id'] = 'first'

                monthly_agg_data = station_data.groupby(['year', 'month']).agg(agg_dict).reset_index()

                # Remove a coluna subbasin_id se existir
                if 'subbasin_id' in monthly_agg_data.columns:
                    monthly_agg_data = monthly_agg_data.drop(columns=['subbasin_id'])

                all_station_data.append(monthly_agg_data)
                
            except Exception as e:
                print(f"  [Error] Failed to process file {filename}: {e}")

    if not all_station_data:
        print("No valid meteorological data found. Exiting.")
        return pd.DataFrame()

    # Concatena os dados de todas as estações em um único DataFrame
    combined_data = pd.concat(all_station_data, ignore_index=True)
    
    print(f"Total processed data shape: {combined_data.shape}")
    print(f"Unique stations found: {combined_data['station_id'].unique().tolist()}")
    
    return combined_data

# A função process_flow_data não é mais necessária, pois os novos arquivos contêm tudo.
# Vamos removê-la ou comentá-la para evitar confusão.
def process_flow_data():
    """Esta função foi descontinuada. Os dados de vazão agora são carregados junto com os dados meteorológicos."""
    print("[Info] process_flow_data is deprecated and no longer in use.")
    return None

def merge_data(flow_data, met_data):
    """Com a nova estrutura, não é necessário merge externo. Apenas filtra por estação/subbacia se necessário."""
    if met_data is None or met_data.empty:
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

    # Split data - 5 last years for testing
    last_year = data_clean['year'].max()
    test_start_year = last_year - 4  # 5 years including the last one
    
    train_data = data_clean[data_clean['year'] < test_start_year]
    test_data = data_clean[data_clean['year'] >= test_start_year]
    print(f"  Train data: {len(train_data)} records, Test data: {len(test_data)} records")
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
    # Carrega e processa todos os dados meteorológicos
    met_data = load_meteorological_data()
    
    if met_data.empty:
        print("No data to process. Exiting.")
        return {}, {}, {}

    # A função merge_data agora apenas separa os dados por estação
    merged_datasets = merge_data(None, met_data)
    
    # Treina modelos para cada estação
    all_results = {}
    all_predictions = {}
    for station_id, data in merged_datasets.items():
        print(f"\n--- Processing Station: {station_id} ---")
        print(f"Station data shape: {data.shape}")
        print(f"Station data columns: {data.columns.tolist()}")
        results, predictions = train_models(data, target_col='flow_next_month')
        if results:
            all_results[station_id] = results
            all_predictions[station_id] = predictions
            print(f"✅ Successfully trained models for station {station_id}")
        else:
            print(f"❌ Could not train models for station {station_id} due to data issues.")
            
    print("\n" + "=" * 50)
    print("Flow Prediction Application Finished")
    return all_results, all_predictions, merged_datasets

if __name__ == "__main__":
    results, predictions, datasets = main()