# Light HydroSense

An AI-powered R&D project that improves hydroelectric water flow predictions using a two-stage machine learning approach. By leveraging meteorological data and advanced ML algorithms, the project enhances monthly flow forecast accuracy with uncertainty quantification, leading to more efficient and sustainable energy generation.

## 🎯 Business Objective

Develop a robust monthly flow prediction system for hydroelectric power generation that:
- Predicts water flow one month ahead using meteorological variables
- Provides uncertainty quantification through confidence intervals
- Supports decision-making for energy generation planning
- Enables risk assessment for hydroelectric operations

## 🏗️ System Architecture

### Two-Stage Machine Learning Approach

#### **Model 1: Primary Flow Prediction**
- **Purpose**: Predict monthly water flow using meteorological predictors
- **Input Variables**: Wind speed (u2_y), Min/Max temperature (tmin_y, tmax_y), Solar radiation (rs_y), Relative humidity (rh_y), Evapotranspiration (eto_y), Precipitation (pr_y)
- **Target**: Monthly flow values for next month
- **Algorithms**: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR

#### **Model 2: Error Prediction & Uncertainty Quantification**
- **Purpose**: Predict residual errors from Model 1 to create confidence intervals
- **Input Variables**: Same meteorological predictors as Model 1
- **Target**: Prediction errors (observed - predicted) from Model 1
- **Output**: Confidence intervals for flow predictions

## 📊 Data Processing Pipeline

### 1. **Data Integration**
- **Meteorological Data**: `glob-funil-subbasin.csv` (monthly aggregated weather data)
- **Flow Data**: `Vazao_FUNIL.xlsx` (daily flow measurements converted to monthly)
- **Spatial Mapping**: 
  - Station 58030000 ↔ Subbasin 24
  - Station 58060000 ↔ Subbasin 36

### 2. **Data Quality Control**
- Remove negative flow values (data quality issues)
- Handle missing values in meteorological data
- Temporal alignment of meteorological and flow data

### 3. **Feature Engineering**
- Monthly aggregation of daily flow data
- Temporal lag implementation (predict month t+1 using month t predictors)
- Feature scaling for algorithm optimization

## 🤖 Machine Learning Implementation

### **Training Strategy**
- **Training Period**: 1998-2015 (18 years)
- **Testing Period**: 2016-2020 (5 years)
- **Cross-Validation**: Time-series aware splitting
- **Feature Scaling**: StandardScaler for linear models

### **Model Selection Criteria**
- **Primary Metrics**: RMSE, MAE, Nash-Sutcliffe Efficiency
- **Secondary Metrics**: Correlation, BIAS
- **Best Performer**: Random Forest (optimal bias-variance tradeoff)

### **Uncertainty Quantification**
- **Method**: Residual-based error modeling
- **Confidence Level**: 95% (configurable)
- **Statistical Foundation**: Normal distribution assumption with Z-score = 1.96

## 📈 Business Performance

### **Station 58030000 (Subbasin 24)**
- **RMSE**: 5.06 m³/s
- **Correlation**: 0.742
- **Nash-Sutcliffe**: 0.346
- **Mean Flow**: 11.18 m³/s

### **Station 58060000 (Subbasin 36)**
- **RMSE**: 2.90 m³/s  
- **Correlation**: 0.653
- **Nash-Sutcliffe**: 0.202
- **Mean Flow**: 6.91 m³/s

## 🚀 Usage

### **Quick Start**
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete analysis
python flow_prediction_app.py

# Generate Model 1 evaluation
python evaluate_model1.py

# Generate Model 2 evaluation and plots
python evaluate_model2.py

# Create visualization
python create_plots.py
```

### **Output Files**
- `model1_performance_metrics.csv` - Detailed Model 1 performance metrics
- `model2_error_prediction_metrics.csv` - Model 2 error prediction metrics
- `flow_prediction_station_*.png` - Time series plots with confidence intervals
- `README_results.md` - Comprehensive results documentation

## 🔧 Technical Stack

- **Python 3.12+**
- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib
- **Statistical Analysis**: scipy
- **Data Processing**: openpyxl

## 📋 Requirements

```
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
openpyxl>=3.1.0
numpy>=1.24.0
scipy>=1.10.0
```

## 🎯 Business Value

### **Operational Benefits**
- **Improved Planning**: 1-month ahead flow predictions enable better resource allocation
- **Risk Management**: Confidence intervals quantify prediction uncertainty
- **Cost Optimization**: Better flow forecasts reduce operational inefficiencies
- **Sustainability**: Enhanced predictions support renewable energy optimization

### **Technical Advantages**
- **Scalable Architecture**: Modular design allows easy extension to new stations
- **Robust Validation**: Time-series cross-validation ensures realistic performance estimates
- **Uncertainty Quantification**: Provides decision-makers with confidence bounds
- **Multiple Algorithms**: Ensemble approach improves prediction reliability

## 📊 Model Interpretability

### **Feature Importance** (Random Forest)
1. **Precipitation (pr_y)**: Primary driver of flow variations
2. **Evapotranspiration (eto_y)**: Seasonal water loss indicator
3. **Temperature (tmin_y, tmax_y)**: Snowmelt and evaporation effects
4. **Solar Radiation (rs_y)**: Energy balance component
5. **Humidity (rh_y)**: Atmospheric moisture content
6. **Wind Speed (u2_y)**: Evaporation enhancement factor

## 🔮 Future Enhancements

- **Deep Learning Models**: LSTM/GRU for temporal sequence modeling
- **Ensemble Methods**: Combine multiple model predictions
- **Real-time Integration**: API for live meteorological data
- **Spatial Modeling**: Incorporate upstream-downstream relationships
- **Climate Change Adaptation**: Long-term trend analysis and adjustment

## 📞 Support

For technical questions or business inquiries regarding the Light HydroSense system, please refer to the detailed results documentation in `README_results.md`.
