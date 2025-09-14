# Flow Prediction Application Results

## Overview
This application implements a two-stage machine learning approach for monthly flow prediction using meteorological data.

## Data Processing
- **Meteorological Data**: `glob-funil-subbasin.csv` with predictor variables (u2_y, tmin_y, tmax_y, rs_y, rh_y, eto_y, pr_y)
- **Flow Data**: `Vazao_FUNIL.xlsx` converted from daily to monthly scale
- **Mapping**: Station 58030000 → Subbasin 24, Station 58060000 → Subbasin 36
- **Data Split**: Training (until 2015), Testing (2016-2020)
- **Filtering**: Negative flow values removed

## Model 1: Flow Prediction
Multiple ML algorithms tested:
- Linear Regression
- Ridge Regression  
- Random Forest
- Gradient Boosting
- Support Vector Regression

### Performance Summary (Test Set):
**Station 58030000 (Subbasin 24):**
- Best Model: Random Forest
- RMSE: 5.06, MAE: 4.08, Correlation: 0.742, Nash-Sutcliffe: 0.346

**Station 58060000 (Subbasin 36):**
- Best Model: Random Forest
- RMSE: 2.90, MAE: 2.17, Correlation: 0.653, Nash-Sutcliffe: 0.202

## Model 2: Error Prediction
Predicts residuals from Model 1 to create confidence intervals.

### Performance Summary (Test Set):
Model 2 shows limited predictive power for errors, indicating that Model 1 captures most of the predictable signal.

## Output Files
1. `model1_performance_metrics.csv` - Detailed Model 1 performance metrics
2. `model2_error_prediction_metrics.csv` - Detailed Model 2 performance metrics  
3. `flow_prediction_station_58030000_subbasin_24.png` - Time series plot with confidence intervals
4. `flow_prediction_station_58060000_subbasin_36.png` - Time series plot with confidence intervals

## Key Findings
- Random Forest and Gradient Boosting perform best for flow prediction
- Model performance is better for training data than test data (expected)
- Station 58030000 shows higher variability and prediction errors
- Confidence intervals provide uncertainty quantification for predictions