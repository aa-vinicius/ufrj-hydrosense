"""
Pytest configuration and shared fixtures for HydroSense tests.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_meteorological_data():
    """Create sample meteorological data for testing (novo padrão)."""
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='M')
    n_records = len(dates) * 2  # Duas sub-bacias
    data = {
        'year': np.repeat([d.year for d in dates], 2),
        'month': np.repeat([d.month for d in dates], 2),
        'subbasin_id': np.tile([24, 36], len(dates)),
        'u2': np.random.uniform(1.0, 3.0, n_records),
        'tmin': np.random.uniform(15.0, 25.0, n_records),
        'tmax': np.random.uniform(25.0, 35.0, n_records),
        'rs': np.random.uniform(15.0, 25.0, n_records),
        'rh': np.random.uniform(60.0, 90.0, n_records),
        'eto': np.random.uniform(3.0, 7.0, n_records),
        'pr': np.random.uniform(0.0, 200.0, n_records),
        'station_id': np.tile([58030000, 58060000], len(dates)),
        'flow_next_month': np.random.uniform(8.0, 20.0, n_records)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='D')
    
    data = {
        'Data': dates,
        58030000: np.random.uniform(5.0, 20.0, len(dates)),
        58060000: np.random.uniform(3.0, 15.0, len(dates))
    }
    
    # Add some negative values to test filtering
    data[58030000][:10] = -999.0
    data[58060000][:5] = -999.0
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_monthly_flow_data():
    """Create sample monthly flow data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2010-01-31', '2020-12-31', freq='M')
    
    data = {
        'Data': dates,
        58030000: np.random.uniform(8.0, 15.0, len(dates)),
        58060000: np.random.uniform(5.0, 12.0, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_merged_data():
    """Create sample merged dataset for testing (novo padrão)."""
    np.random.seed(42)
    n_records = 100
    subbasins = np.random.choice([24, 36], n_records)
    station_ids = [58030000 if sb == 24 else 58060000 for sb in subbasins]
    data = {
        'year': np.random.randint(2010, 2021, n_records),
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

@pytest.fixture
def sample_predictions():
    """Create sample model predictions for testing (novo padrão)."""
    np.random.seed(42)
    n_samples = 50
    return {
        'Random_Forest': {
            'train_pred': np.random.uniform(5.0, 15.0, n_samples),
            'test_pred': np.random.uniform(5.0, 15.0, n_samples//2),
            'y_train': np.random.uniform(5.0, 15.0, n_samples),
            'y_test': np.random.uniform(5.0, 15.0, n_samples//2),
            'train_dates': pd.DataFrame({
                'year': np.random.randint(2010, 2016, n_samples),
                'month': np.random.randint(1, 13, n_samples)
            }),
            'test_dates': pd.DataFrame({
                'year': np.random.randint(2016, 2021, n_samples//2),
                'month': np.random.randint(1, 13, n_samples//2)
            })
        }
    }

@pytest.fixture
def mock_file_paths(tmp_path):
    """Create temporary file paths for testing."""
    return {
        'met_data': tmp_path / "met_data.csv",
        'flow_data': tmp_path / "flow_data.xlsx",
        'output_csv': tmp_path / "output.csv",
        'output_png': tmp_path / "output.png"
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
