import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model2_error_prediction import main_model2
import os

def create_time_series_plots():
    """Create time series plots with confidence intervals for all stations."""
    print("Creating time series plots with confidence intervals...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    model1_results, model1_preds, model2_results, model2_preds, ci, datasets = main_model2()
    
    if not model1_preds:
        print("No model predictions available to create plots.")
        return

    for station_id in model1_preds.keys():
        print(f"\nGenerating plot for Station ID: {station_id}")

        subbasin_id = None
        if datasets and station_id in datasets and 'subbasin_id' in datasets[station_id].columns:
            subbasin_id = datasets[station_id]['subbasin_id'].iloc[0]
        
        if 'Random_Forest' not in model1_preds[station_id]:
            print(f"  [Warning] 'Random_Forest' model not found for station {station_id}. Skipping plot.")
            continue

        test_dates = model1_preds[station_id]['Random_Forest']['test_dates']
        observed = model1_preds[station_id]['Random_Forest']['y_test']
        model1_pred = model1_preds[station_id]['Random_Forest']['test_pred']

        date_df = test_dates[['year', 'month']].copy()
        date_df['day'] = 1
        dates = pd.to_datetime(date_df.assign(day=1)[['year', 'month', 'day']].astype(int).apply(lambda row: f"{row['year']}-{row['month']:02d}-01", axis=1))
        
        if ci and station_id in ci:
            corrected_pred = ci[station_id]['corrected_predictions']
            lower_bound = ci[station_id]['lower_bound']
            upper_bound = ci[station_id]['upper_bound']
        else:
            corrected_pred = model1_pred
            error_std = np.std(observed - model1_pred)
            lower_bound = model1_pred - 1.96 * error_std
            upper_bound = model1_pred + 1.96 * error_std
        
        plt.figure(figsize=(15, 8))
        plt.plot(dates, observed, 'b-', label='Observed', linewidth=2, alpha=0.8)
        plt.plot(dates, model1_pred, 'r--', label='Model 1 Prediction', linewidth=2, alpha=0.7)
        plt.plot(dates, corrected_pred, 'g-', label='Corrected Prediction (Model 1 + Error Model)', linewidth=2, alpha=0.8)
        plt.fill_between(dates, lower_bound, upper_bound, alpha=0.3, color='gray', label='95% Confidence Interval')
        
        title = f'Flow Prediction - Station {station_id}'
        if subbasin_id:
            title += f' (Subbasin {int(subbasin_id)})'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Flow (m³/s)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'flow_prediction_station_{station_id}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Plot saved as '{filename}'")
        
        plt.close()
    
    print("\nAll plots created successfully!")

if __name__ == "__main__":
    create_time_series_plots()