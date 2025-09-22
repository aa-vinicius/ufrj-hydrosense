import pandas as pd
import numpy as np
from model2_error_prediction import main_model2
import os

def save_model2_results():
    """Evaluate Model 2 and save results to CSV"""
    print("Evaluating Model 2 performance...")
    
    # Run Model 2
    model1_results, model1_preds, model2_results, model2_preds, ci, datasets = main_model2()
    
    # Prepare Model 2 results for CSV
    model2_results_list = []
    
    if not model2_results:
        print("No Model 2 results to process.")
        return model1_results, model1_preds, model2_results, model2_preds, ci, datasets

    for station_id in model2_results.keys():
        # Tenta obter o subbasin_id dos dados, se disponível
        subbasin_id = None
        if datasets and station_id in datasets and 'subbasin_id' in datasets[station_id].columns:
            subbasin_id = datasets[station_id]['subbasin_id'].iloc[0]

        for model_name, metrics in model2_results[station_id].items():
            # Training results
            train_row = {
                'Flow_Station': station_id,
                'Subbasin_ID': subbasin_id,
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
            model2_results_list.append(train_row)
            
            # Test results
            test_row = {
                'Flow_Station': station_id,
                'Subbasin_ID': subbasin_id,
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
            model2_results_list.append(test_row)
    
    # Convert to DataFrame and save
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    if not model2_results_list:
        print("No data to save for Model 2.")
    else:
        model2_df = pd.DataFrame(model2_results_list)
        model2_df.to_csv(os.path.join(output_dir, 'model2_error_prediction_metrics.csv'), index=False)
        
        print("Model 2 results saved to 'outputs/model2_error_prediction_metrics.csv'")
        print("\nModel 2 Performance Summary:")
        print(model2_df.groupby(['Flow_Station', 'Dataset'])[['RMSE', 'MAE', 'Bias', 'Nash_Sutcliffe']].mean())
    
    return model1_results, model1_preds, model2_results, model2_preds, ci, datasets

if __name__ == "__main__":
    results = save_model2_results()