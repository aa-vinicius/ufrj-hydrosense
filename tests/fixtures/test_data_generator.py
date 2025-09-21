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
    
    def generate_meteo_vazao_csv(self, filepath, start_year=2010, end_year=2020, subbasins=[24, 36]):
        """Gera CSV no novo formato para o pipeline revisado."""
        dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='M')
        data = []
        for date in dates:
            for subbasin in subbasins:
                station_id = 58030000 if subbasin == 24 else 58060000
                seasonal_factor = np.sin(2 * np.pi * date.month / 12)
                record = {
                    'year': date.year,
                    'month': date.month,
                    'subbasin_id': subbasin,
                    'u2': np.random.uniform(1.5, 2.5) + 0.3 * seasonal_factor,
                    'tmin': 20 + 5 * seasonal_factor + np.random.normal(0, 2),
                    'tmax': 30 + 5 * seasonal_factor + np.random.normal(0, 2),
                    'rs': 20 + 3 * seasonal_factor + np.random.normal(0, 1),
                    'rh': 75 - 10 * seasonal_factor + np.random.normal(0, 5),
                    'eto': 5 + 2 * seasonal_factor + np.random.normal(0, 0.5),
                    'pr': max(0, 100 + 50 * seasonal_factor + np.random.normal(0, 30)),
                    'station_id': station_id,
                    'flow_next_month': 10 + 2 * seasonal_factor + np.random.normal(0, 1)
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
        """Gera um pequeno DataFrame no novo formato do pipeline revisado."""
        np.random.seed(self.seed)
        subbasins = np.random.choice([24, 36], n_records)
        station_ids = [58030000 if sb == 24 else 58060000 for sb in subbasins]
        data = {
            'year': np.random.randint(2015, 2021, n_records),
            'month': np.random.randint(1, 13, n_records),
            'subbasin_id': subbasins,
            'u2': np.random.uniform(1.5, 2.5, n_records),
            'tmin': np.random.uniform(15.0, 25.0, n_records),
            'tmax': np.random.uniform(25.0, 35.0, n_records),
            'rs': np.random.uniform(15.0, 25.0, n_records),
            'rh': np.random.uniform(60.0, 90.0, n_records),
            'eto': np.random.uniform(3.0, 7.0, n_records),
            'pr': np.random.uniform(0.0, 200.0, n_records),
            'station_id': station_ids,
            'flow_next_month': np.random.uniform(8.0, 15.0, n_records)
        }
        return pd.DataFrame(data)
    
    def generate_edge_case_data(self):
        """Gera DataFrames de casos extremos no novo formato do pipeline revisado."""
        # Empty dataset
        empty_df = pd.DataFrame({col: [] for col in [
            'year', 'month', 'subbasin_id', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr', 'station_id', 'flow_next_month']})
        # Single record dataset
        single_record = pd.DataFrame({
            'year': [2015],
            'month': [6],
            'subbasin_id': [24],
            'u2': [2.0],
            'tmin': [20.0],
            'tmax': [30.0],
            'rs': [20.0],
            'rh': [75.0],
            'eto': [5.0],
            'pr': [100.0],
            'station_id': [58030000],
            'flow_next_month': [10.0]
        })
        # Dataset with missing values
        missing_data = self.generate_small_dataset(20)
        missing_data.loc[0:5, 'u2'] = np.nan
        missing_data.loc[10:15, 'flow_next_month'] = np.nan
        # Dataset with extreme values
        extreme_data = self.generate_small_dataset(20)
        extreme_data.loc[0, 'pr'] = 1000.0  # Precipitação extrema
        extreme_data.loc[1, 'tmin'] = -10.0  # Temperatura extrema
        extreme_data.loc[2, 'flow_next_month'] = 100.0  # Vazão extrema
        return {
            'empty': empty_df,
            'single_record': single_record,
            'missing_values': missing_data,
            'extreme_values': extreme_data
        }

def create_test_files(temp_dir):
    """Cria arquivos de teste no novo formato para o pipeline revisado."""
    generator = TestDataGenerator()
    data_dir = temp_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    met_file = data_dir / 'meteo_vazao_shifted_station_58030000.csv'
    met_data = generator.generate_meteo_vazao_csv(met_file)
    return {
        'met_file': met_file,
        'met_data': met_data
    }
