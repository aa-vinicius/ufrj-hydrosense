import pandas as pd
import numpy as np
from flow_prediction_app import main

def save_model1_results():
    """Evaluate Model 1 and save results to CSV"""
    print("Evaluating Model 1 performance...")
    
    # Run the main application
    results, predictions, datasets = main()
    
    # Prepare results for CSV
    model1_results = []
    
    for flow_col in results.keys():
        subbasin_id = 24 if flow_col == 58030000 else 36
        
        for model_name, metrics in results[flow_col].items():
            # Training results
            train_row = {
                'Flow_Station': flow_col,
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
            model1_results.append(train_row)
            
            # Test results
            test_row = {
                'Flow_Station': flow_col,
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
            model1_results.append(test_row)
    
    # Convert to DataFrame and save
    model1_df = pd.DataFrame(model1_results)
    model1_df.to_csv('../outputs/model1_performance_metrics.csv', index=False)
    
    print("Model 1 results saved to 'model1_performance_metrics.csv'")
    print("\nModel 1 Performance Summary:")
    print(model1_df.groupby(['Flow_Station', 'Dataset'])[['RMSE', 'MAE', 'Bias', 'Nash_Sutcliffe']].mean())
    
    return results, predictions, datasets

if __name__ == "__main__":
    results, predictions, datasets = save_model1_results()