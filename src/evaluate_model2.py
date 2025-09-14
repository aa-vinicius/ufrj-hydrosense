import pandas as pd
import numpy as np
from model2_error_prediction import main_model2

def save_model2_results():
    """Evaluate Model 2 and save results to CSV"""
    print("Evaluating Model 2 performance...")
    
    # Run Model 2
    model1_results, model1_preds, model2_results, model2_preds, ci, datasets = main_model2()
    
    # Prepare Model 2 results for CSV
    model2_results_list = []
    
    for flow_col in model2_results.keys():
        subbasin_id = 24 if flow_col == 58030000 else 36
        
        for model_name, metrics in model2_results[flow_col].items():
            # Training results
            train_row = {
                'Flow_Station': flow_col,
                'Subbasin_ID': subbasin_id,
                'Model': model_name,
                'Dataset': 'Training',
                'RMSE': metrics['train']['RMSE'],
                'MAE': metrics['train']['MAE'],
                'Correlation': metrics['train']['Correlation'],
                'BIAS': metrics['train']['BIAS'],
                'Nash_Sutcliffe': metrics['train']['Nash_Sutcliffe']
            }
            model2_results_list.append(train_row)
            
            # Test results
            test_row = {
                'Flow_Station': flow_col,
                'Subbasin_ID': subbasin_id,
                'Model': model_name,
                'Dataset': 'Test',
                'RMSE': metrics['test']['RMSE'],
                'MAE': metrics['test']['MAE'],
                'Correlation': metrics['test']['Correlation'],
                'BIAS': metrics['test']['BIAS'],
                'Nash_Sutcliffe': metrics['test']['Nash_Sutcliffe']
            }
            model2_results_list.append(test_row)
    
    # Convert to DataFrame and save
    model2_df = pd.DataFrame(model2_results_list)
    model2_df.to_csv('../outputs/model2_error_prediction_metrics.csv', index=False)
    
    print("Model 2 results saved to 'model2_error_prediction_metrics.csv'")
    print("\nModel 2 Performance Summary:")
    print(model2_df.groupby(['Flow_Station', 'Dataset'])[['RMSE', 'MAE', 'Correlation', 'Nash_Sutcliffe']].mean())
    
    return model1_results, model1_preds, model2_results, model2_preds, ci, datasets

if __name__ == "__main__":
    results = save_model2_results()