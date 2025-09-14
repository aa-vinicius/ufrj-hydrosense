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

## 📁 Project Structure

```
light-hydrosense/
├── 📁 src/                          # Source code
│   ├── flow_prediction_app.py       # Main Model 1 implementation
│   ├── model2_error_prediction.py   # Model 2 error prediction
│   ├── evaluate_model1.py           # Model 1 evaluation
│   ├── evaluate_model2.py           # Model 2 evaluation
│   ├── create_plots.py              # Visualization generation
│   └── __init__.py                  # Package initialization
├── 📁 data/                         # Input datasets
│   ├── glob-funil-subbasin.csv      # Meteorological data
│   └── Vazao_FUNIL.xlsx             # Flow measurements
├── 📁 outputs/                      # Generated results
│   ├── model1_performance_metrics.csv
│   ├── model2_error_prediction_metrics.csv
│   ├── flow_prediction_station_58030000_subbasin_24.png
│   └── flow_prediction_station_58060000_subbasin_36.png
├── 📁 docs/                         # Documentation
│   └── README_results.md            # Detailed results analysis
├── 🚀 run_analysis.py               # Main execution script
├── 📋 requirements.txt              # Dependencies
└── 📖 README.md                     # This file
```

## 🚀 Usage

### **Quick Start**
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete analysis (recommended)
python run_analysis.py
```

### **Individual Script Execution**
```bash
# From project root, run individual components:
cd src

# Model 1: Flow prediction
python flow_prediction_app.py

# Model 1 evaluation
python evaluate_model1.py

# Model 2: Error prediction and evaluation
python evaluate_model2.py

# Generate visualizations
python create_plots.py
```

### **Output Files**
- `outputs/model1_performance_metrics.csv` - Detailed Model 1 performance metrics
- `outputs/model2_error_prediction_metrics.csv` - Model 2 error prediction metrics
- `outputs/flow_prediction_station_*.png` - Time series plots with confidence intervals
- `docs/README_results.md` - Comprehensive results documentation

## 🔧 Technical Stack

- **Python 3.12+**
- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib
- **Statistical Analysis**: scipy
- **Data Processing**: openpyxl
- **Testing**: pytest, pytest-cov
- **Development**: black, flake8, mypy (optional)

## 📋 Requirements

### **Production Requirements**
```bash
pip install -r requirements.txt
```

### **Development Requirements** 
```bash
pip install -r requirements-dev.txt
```

### **Core Dependencies**
```
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
openpyxl>=3.1.0
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.0.0
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

## 🧪 Testing

Light HydroSense includes a comprehensive test suite following Test-Driven Development (TDD) principles.

### **Test Structure**
```
tests/
├── 📁 unit/                    # Unit tests for individual functions
│   ├── test_flow_prediction_app.py
│   ├── test_model2_error_prediction.py
│   ├── test_evaluation_methods.py
│   └── test_plotting_methods.py
├── 📁 integration/             # End-to-end pipeline tests
│   └── test_complete_pipeline.py
├── 📁 fixtures/                # Test data generators
│   └── test_data_generator.py
├── conftest.py                 # Shared test fixtures
└── test_markers.py             # Test categorization examples
```

### **Running Tests**

#### **Quick Test Commands**
```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit                    # Unit tests only
python run_tests.py --integration             # Integration tests only
python run_tests.py --category model1         # Model 1 tests
python run_tests.py --category model2         # Model 2 tests
python run_tests.py --coverage                # Tests with coverage report
```

#### **Advanced Testing Options**
```bash
# Performance and scalability tests
python run_tests.py --performance

# Validate test structure
python run_tests.py --validate

# Check dependencies
python run_tests.py --check-deps

# Run specific test file
pytest tests/unit/test_flow_prediction_app.py -v

# Run tests with specific markers
pytest -m "fast and unit" -v
pytest -m "slow or integration" -v
```

### **Test Categories (Markers)**
- `unit` - Unit tests for individual functions
- `integration` - End-to-end pipeline tests  
- `fast` - Quick tests (< 1 second)
- `slow` - Longer tests (> 1 second)
- `model1` - Flow prediction model tests
- `model2` - Error prediction model tests
- `plotting` - Visualization tests
- `evaluation` - Performance evaluation tests
- `pipeline` - Complete workflow tests

### **Coverage Reports**
```bash
# Generate HTML coverage report
python run_tests.py --coverage

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### **Test Development**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests during development
pytest tests/ -v --tb=short

# Run tests with file watching (requires pytest-watch)
ptw tests/ src/
```

## 🔧 Development

### **Adding New Features**
1. **Write tests first** (TDD approach)
   ```bash
   # Create test file
   touch tests/unit/test_new_feature.py
   
   # Write failing tests
   # Implement feature to make tests pass
   ```

2. Create new modules in `src/` directory
3. Update `run_analysis.py` if new scripts need to be executed
4. Ensure outputs are saved to `outputs/` directory
5. Update documentation in `docs/` directory
6. **Run full test suite** to ensure no regressions

### **Testing Changes**
```bash
# Test individual components
cd src && python <script_name>.py

# Test complete pipeline
python run_analysis.py

# Run relevant tests
python run_tests.py --category <relevant_category>

# Run full test suite
python run_tests.py --all
```

### **Code Quality**
```bash
# Format code (if black is installed)
black src/ tests/

# Check code style (if flake8 is installed)
flake8 src/ tests/

# Type checking (if mypy is installed)
mypy src/
```

## 📞 Support

For technical questions or business inquiries regarding the Light HydroSense system, please refer to the detailed results documentation in `docs/README_results.md`.
