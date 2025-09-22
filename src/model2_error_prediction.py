import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from flow_prediction_app import calculate_metrics, train_models, main
import warnings
warnings.filterwarnings('ignore')

def train_error_models(data, target_col, model1_predictions, train_year_cutoff=2019):
    """Treina modelos para prever os erros do Modelo 1 usando a nova estrutura de dados"""
    predictor_cols = ['year', 'month', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
    if data is None or data.empty or not all(col in data.columns for col in predictor_cols):
        print(f"Warning: DataFrame vazio ou colunas ausentes para {target_col}")
        return None, None
    # Split data - flexível baseado nos dados disponíveis
    train_data = data[data['year'] <= train_year_cutoff]
    test_data = data[data['year'] > train_year_cutoff]
    if len(train_data) == 0 or len(test_data) == 0:
        print(f"Warning: Insufficient data for {target_col}")
        return None, None
    X_train = train_data[predictor_cols]
    X_test = test_data[predictor_cols]
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
    best_model = 'Random_Forest'
    if best_model not in model1_predictions:
        print(f"Warning: {best_model} not found in Model 1 predictions")
        return None, None
    y_train = model1_predictions[best_model]['y_train']
    y_train_pred = model1_predictions[best_model]['train_pred']
    y_test = model1_predictions[best_model]['y_test']
    y_test_pred = model1_predictions[best_model]['test_pred']
    # Checar tamanhos e NaN
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        print("Warning: Tamanhos inconsistentes entre X e y para erro.")
        return None, None
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)) or np.any(np.isnan(y_train_pred)) or np.any(np.isnan(y_test_pred)):
        print("Warning: Dados com NaN detectados para erro.")
        return None, None
    train_errors = y_train - y_train_pred
    test_errors = y_test - y_test_pred
    for model_name, model in models.items():
        print(f"Training {model_name} for error prediction of {target_col}...")
        if model_name in ['Linear_Regression', 'Ridge', 'SVR', 'MLP']:
            model.fit(X_train_scaled, train_errors)
            train_error_pred = model.predict(X_train_scaled)
            test_error_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, train_errors)
            train_error_pred = model.predict(X_train)
            test_error_pred = model.predict(X_test)
        train_metrics = calculate_metrics(train_errors, train_error_pred)
        test_metrics = calculate_metrics(test_errors, test_error_pred)
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        predictions[model_name] = {
            'train_error_pred': train_error_pred,
            'test_error_pred': test_error_pred,
            'train_errors': train_errors,
            'test_errors': test_errors,
            'train_dates': train_data[['year', 'month']],
            'test_dates': test_data[['year', 'month']]
        }
    return results, predictions

def create_confidence_intervals(model1_preds, model2_preds, confidence_level=0.95):
    """Create confidence intervals using Model 2 error predictions"""
    # Use the best error prediction model (Random Forest)
    best_error_model = 'Random_Forest'
    
    if best_error_model not in model2_preds:
        print(f"Warning: {best_error_model} not found in Model 2 predictions")
        return None
    
    # Get Model 1 predictions and Model 2 error predictions
    test_flow_pred = model1_preds['Random_Forest']['test_pred']
    test_error_pred = model2_preds[best_error_model]['test_error_pred']
    # Checar tamanhos
    if len(test_flow_pred) != len(test_error_pred):
        print("Warning: Tamanhos inconsistentes entre test_flow_pred e test_error_pred.")
        return None
    
    # Calculate standard deviation of error predictions for confidence intervals
    error_std = np.std(model2_preds[best_error_model]['test_errors'])
    
    # Z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Create confidence intervals
    lower_bound = test_flow_pred - test_error_pred - z_score * error_std
    upper_bound = test_flow_pred - test_error_pred + z_score * error_std
    
    # Corrected predictions (Model 1 - Model 2 error prediction)
    corrected_pred = test_flow_pred - test_error_pred
    
    return {
        'corrected_predictions': corrected_pred,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'original_predictions': test_flow_pred,
        'error_predictions': test_error_pred
    }

def main_model2():
    """Main function for Model 2"""
    print("Starting Model 2: Error Prediction")
    print("=" * 50)
    
    # Get Model 1 results
    model1_results, model1_predictions, datasets = main()
    
    # Train Model 2 for each flow station
    model2_results = {}
    model2_predictions = {}
    confidence_intervals = {}
    
    for flow_col in model1_results.keys():
        print(f"\n{'='*20} Model 2 for {flow_col} {'='*20}")
        
        if flow_col in datasets and flow_col in model1_predictions:
            error_results, error_predictions = train_error_models(
                datasets[flow_col], flow_col, model1_predictions[flow_col]
            )
            
            if error_results:
                model2_results[flow_col] = error_results
                model2_predictions[flow_col] = error_predictions
                
                # Create confidence intervals
                ci = create_confidence_intervals(
                    model1_predictions[flow_col], 
                    error_predictions
                )
                if ci:
                    confidence_intervals[flow_col] = ci
    
    return model1_results, model1_predictions, model2_results, model2_predictions, confidence_intervals, datasets

if __name__ == "__main__":
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy for confidence intervals...")
        import subprocess
        subprocess.check_call(['./venv/bin/pip', 'install', 'scipy'])
        from scipy import stats
    
    model1_results, model1_preds, model2_results, model2_preds, ci, datasets = main_model2()