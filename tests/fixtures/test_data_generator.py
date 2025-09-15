"""
Test data generators for HydroSense testing.
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TestDataGenerator:
    """Generate realistic test data for various testing scenarios."""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible tests."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_meteorological_csv(self, filepath, start_year=2010, end_year=2020, subbasins=[24, 36]):
        """Generate a realistic meteorological CSV file."""
        dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='M')
        
        data = []
        for date in dates:
            for subbasin in subbasins:
                # Generate realistic seasonal patterns
                month = date.month
                seasonal_factor = np.sin(2 * np.pi * month / 12)
                
                record = {
                    'year_x': date.year,
                    'month_x': date.month,
                    'ID_Subbasin': subbasin,
                    'u2_y': np.random.uniform(1.5, 2.5) + 0.3 * seasonal_factor,
                    'tmin_y': 20 + 5 * seasonal_factor + np.random.normal(0, 2),
                    'tmax_y': 30 + 5 * seasonal_factor + np.random.normal(0, 2),
                    'rs_y': 20 + 3 * seasonal_factor + np.random.normal(0, 1),
                    'rh_y': 75 - 10 * seasonal_factor + np.random.normal(0, 5),
                    'eto_y': 5 + 2 * seasonal_factor + np.random.normal(0, 0.5),
                    'pr_y': max(0, 100 + 50 * seasonal_factor + np.random.normal(0, 30))
                }
                data.append(record)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return df
    
    def generate_flow_excel(self, filepath, start_year=2010, end_year=2020):
        """Generate a realistic flow Excel file."""
        dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
        
        # Generate flow data with seasonal patterns and some noise
        flow_58030000 = []
        flow_58060000 = []
        
        for date in dates:
            # Seasonal pattern
            day_of_year = date.timetuple().tm_yday
            seasonal = np.sin(2 * np.pi * day_of_year / 365)
            
            # Base flow with seasonal variation
            flow_1 = 12 + 4 * seasonal + np.random.normal(0, 2)
            flow_2 = 8 + 3 * seasonal + np.random.normal(0, 1.5)
            
            # Ensure positive values (add some negative for testing filtering)
            if np.random.random() < 0.01:  # 1% chance of negative (data quality issue)
                flow_1 = -999.0
            if np.random.random() < 0.005:  # 0.5% chance of negative
                flow_2 = -999.0
                
            flow_58030000.append(max(0.1, flow_1) if flow_1 > 0 else flow_1)
            flow_58060000.append(max(0.1, flow_2) if flow_2 > 0 else flow_2)
        
        data = {
            'Data': dates,
            58030000: flow_58030000,
            58060000: flow_58060000
        }
        
        df = pd.DataFrame(data)
        
        # Create Excel file with specific sheet name
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Vazao_FUNIL', index=False)
        
        return df
    
    def generate_small_dataset(self, n_records=50):
        """Generate a small dataset for quick testing."""
        np.random.seed(self.seed)
        
        data = {
            'year_x': np.random.randint(2015, 2021, n_records),
            'month_x': np.random.randint(1, 13, n_records),
            'ID_Subbasin': np.random.choice([24, 36], n_records),
            'u2_y': np.random.uniform(1.0, 3.0, n_records),
            'tmin_y': np.random.uniform(15.0, 25.0, n_records),
            'tmax_y': np.random.uniform(25.0, 35.0, n_records),
            'rs_y': np.random.uniform(15.0, 25.0, n_records),
            'rh_y': np.random.uniform(60.0, 90.0, n_records),
            'eto_y': np.random.uniform(3.0, 7.0, n_records),
            'pr_y': np.random.uniform(0.0, 200.0, n_records),
            58030000: np.random.uniform(8.0, 15.0, n_records),
            58060000: np.random.uniform(5.0, 12.0, n_records)
        }
        
        return pd.DataFrame(data)
    
    def generate_edge_case_data(self):
        """Generate edge case data for robust testing."""
        # Empty dataset
        empty_df = pd.DataFrame()
        
        # Single record dataset
        single_record = pd.DataFrame({
            'year_x': [2015],
            'month_x': [6],
            'ID_Subbasin': [24],
            'u2_y': [2.0],
            'tmin_y': [20.0],
            'tmax_y': [30.0],
            'rs_y': [20.0],
            'rh_y': [75.0],
            'eto_y': [5.0],
            'pr_y': [100.0],
            58030000: [10.0],
            58060000: [7.0]
        })
        
        # Dataset with missing values
        missing_data = self.generate_small_dataset(20)
        missing_data.loc[0:5, 'u2_y'] = np.nan
        missing_data.loc[10:15, 58030000] = np.nan
        
        # Dataset with extreme values
        extreme_data = self.generate_small_dataset(20)
        extreme_data.loc[0, 'pr_y'] = 1000.0  # Extreme precipitation
        extreme_data.loc[1, 'tmin_y'] = -10.0  # Extreme temperature
        extreme_data.loc[2, 58030000] = 100.0  # Extreme flow
        
        return {
            'empty': empty_df,
            'single_record': single_record,
            'missing_values': missing_data,
            'extreme_values': extreme_data
        }

def create_test_files(temp_dir):
    """Create all necessary test files in a temporary directory."""
    generator = TestDataGenerator()
    
    # Create data subdirectory
    data_dir = temp_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Generate test files
    met_file = data_dir / 'glob-funil-subbasin.csv'
    flow_file = data_dir / 'Vazao_FUNIL.xlsx'
    
    met_data = generator.generate_meteorological_csv(met_file)
    flow_data = generator.generate_flow_excel(flow_file)
    
    return {
        'met_file': met_file,
        'flow_file': flow_file,
        'met_data': met_data,
        'flow_data': flow_data
    }
