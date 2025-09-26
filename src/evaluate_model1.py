import pandas as pd
import numpy as np
from flow_prediction_app import main
import os

def save_model1_results():
    """Evaluate Model 1 and save results to CSV"""
    print("Evaluating Model 1 performance...")
    
    # Run the main application
    results, predictions, datasets = main()
    
    # Prepare results for CSV
    model1_results = []
    
    for station_id in results.keys():
        # Tenta obter o subbasin_id dos dados, se disponível
        for model_name, metrics in results[station_id].items():
            # Training results
            train_row = {
                'station_id': station_id,
                'Model': model_name,
                'Dataset': 'Training',
                'RMSE': metrics['train']['RMSE'],
                'MAE': metrics['train']['MAE'],
                'R2': metrics['train'].get('R2', None),
                'KGE': metrics['train'].get('KGE', None),
                'PBIAS': metrics['train'].get('PBIAS', None),
                'Bias': metrics['train'].get('Bias', metrics['train'].get('BIAS', None)),
                'Nash_Sutcliffe': metrics['train']['Nash_Sutcliffe']
            }
            model1_results.append(train_row)
            # Test results
            test_row = {
                'station_id': station_id,
                'Model': model_name,
                'Dataset': 'Test',
                'RMSE': metrics['test']['RMSE'],
                'MAE': metrics['test']['MAE'],
                'R2': metrics['test'].get('R2', None),
                'KGE': metrics['test'].get('KGE', None),
                'PBIAS': metrics['test'].get('PBIAS', None),
                'Bias': metrics['test'].get('Bias', metrics['test'].get('BIAS', None)),
                'Nash_Sutcliffe': metrics['test']['Nash_Sutcliffe']
            }
            model1_results.append(test_row)
    
    # Convert to DataFrame and save
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    model1_df = pd.DataFrame(model1_results)
    model1_df.to_csv(os.path.join(output_dir, 'model1_performance_metrics.csv'), index=False)
    
    print("Model 1 results saved to 'outputs/model1_performance_metrics.csv'")
    print("\nModel 1 Performance Summary:")
    # Verifica quais colunas existem no DataFrame antes de agrupar
    if 'station_id' in model1_df.columns:
        print(model1_df.groupby(['station_id', 'Dataset'])[['RMSE', 'MAE', 'Bias', 'Nash_Sutcliffe']].mean())
    else:
        print("Available columns:", model1_df.columns.tolist())
        print(model1_df.head())
    
    return results, predictions, datasets

if __name__ == "__main__":
    results, predictions, datasets = save_model1_results()