import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model2_error_prediction import main_model2

def create_time_series_plots():
    """Create time series plots with confidence intervals"""
    print("Creating time series plots with confidence intervals...")
    
    # Get all results
    model1_results, model1_preds, model2_results, model2_preds, ci, datasets = main_model2()
    
    # Create plots for each flow station
    for flow_col in model1_preds.keys():
        subbasin_id = 24 if flow_col == 58030000 else 36
        
        # Get test data
        test_dates = model1_preds[flow_col]['Random_Forest']['test_dates']
        observed = model1_preds[flow_col]['Random_Forest']['y_test'].values
        model1_pred = model1_preds[flow_col]['Random_Forest']['test_pred']
        
        # Create date column for plotting
        date_df = test_dates[['year_x', 'month_x']].copy()
        date_df['day'] = 1
        dates = pd.to_datetime(date_df.rename(columns={'year_x': 'year', 'month_x': 'month'}))
        
        # Get confidence intervals if available
        if flow_col in ci:
            corrected_pred = ci[flow_col]['corrected_predictions']
            lower_bound = ci[flow_col]['lower_bound']
            upper_bound = ci[flow_col]['upper_bound']
        else:
            corrected_pred = model1_pred
            # Create simple confidence intervals based on prediction error
            error_std = np.std(observed - model1_pred)
            lower_bound = model1_pred - 1.96 * error_std
            upper_bound = model1_pred + 1.96 * error_std
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot observed data
        plt.plot(dates, observed, 'b-', label='Observed', linewidth=2, alpha=0.8)
        
        # Plot Model 1 predictions
        plt.plot(dates, model1_pred, 'r--', label='Model 1 Prediction', linewidth=2, alpha=0.7)
        
        # Plot corrected predictions (Model 1 - Model 2 error)
        plt.plot(dates, corrected_pred, 'g-', label='Corrected Prediction (Model 1 - Model 2)', linewidth=2, alpha=0.8)
        
        # Plot confidence intervals
        plt.fill_between(dates, lower_bound, upper_bound, alpha=0.3, color='gray', label='95% Confidence Interval')
        
        # Formatting
        plt.title(f'Flow Prediction - Station {flow_col} (Subbasin {subbasin_id})', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Flow (m³/s)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        filename = f'flow_prediction_station_{flow_col}_subbasin_{subbasin_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        
        # Show basic statistics
        print(f"\nStation {flow_col} Statistics:")
        print(f"  Observed mean: {np.mean(observed):.2f}")
        print(f"  Model 1 RMSE: {np.sqrt(np.mean((observed - model1_pred)**2)):.2f}")
        print(f"  Corrected RMSE: {np.sqrt(np.mean((observed - corrected_pred)**2)):.2f}")
        print(f"  Model 1 Correlation: {np.corrcoef(observed, model1_pred)[0,1]:.3f}")
        print(f"  Corrected Correlation: {np.corrcoef(observed, corrected_pred)[0,1]:.3f}")
        
        plt.close()  # Close the figure to free memory
    
    print("\nAll plots created successfully!")

if __name__ == "__main__":
    create_time_series_plots()