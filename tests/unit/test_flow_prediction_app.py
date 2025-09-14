"""
Unit tests for flow_prediction_app.py

Tests all functions and methods in the main flow prediction application
using Test-Driven Development (TDD) approach.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import flow_prediction_app as fpa

class TestMetricsCalculation:
    """Test metric calculation functions."""
    
    def test_nash_sutcliffe_efficiency_perfect_prediction(self):
        """Test NSE with perfect predictions (should equal 1.0)."""
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        
        nse = fpa.nash_sutcliffe_efficiency(observed, predicted)
        
        assert nse == pytest.approx(1.0, abs=1e-10)
    
    def test_nash_sutcliffe_efficiency_mean_prediction(self):
        """Test NSE when predictions equal observed mean (should equal 0.0)."""
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([3, 3, 3, 3, 3])  # Mean of observed
        
        nse = fpa.nash_sutcliffe_efficiency(observed, predicted)
        
        assert nse == pytest.approx(0.0, abs=1e-10)
    
    def test_nash_sutcliffe_efficiency_poor_prediction(self):
        """Test NSE with poor predictions (should be negative)."""
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([10, 10, 10, 10, 10])  # Very poor predictions
        
        nse = fpa.nash_sutcliffe_efficiency(observed, predicted)
        
        assert nse < 0
    
    def test_calculate_metrics_comprehensive(self):
        """Test comprehensive metrics calculation."""
        observed = np.array([10, 15, 12, 18, 14])
        predicted = np.array([11, 14, 13, 17, 15])
        
        metrics = fpa.calculate_metrics(observed, predicted)
        
        # Check all required metrics are present
        required_metrics = ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Check reasonable values
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0
        assert -1 <= metrics['Correlation'] <= 1
        assert metrics['Nash_Sutcliffe'] <= 1
    
    def test_calculate_metrics_edge_cases(self):
        """Test metrics calculation with edge cases."""
        # Single value
        observed = np.array([10])
        predicted = np.array([10])
        
        metrics = fpa.calculate_metrics(observed, predicted)
        assert metrics['RMSE'] == 0
        assert metrics['MAE'] == 0
        assert metrics['BIAS'] == 0

class TestDataProcessing:
    """Test data processing functions."""
    
    @patch('flow_prediction_app.pd.read_excel')
    def test_process_flow_data_basic(self, mock_read_excel, sample_flow_data):
        """Test basic flow data processing."""
        mock_read_excel.return_value = sample_flow_data
        
        result = fpa.process_flow_data()
        
        # Check function was called correctly
        mock_read_excel.assert_called_once_with('../data/Vazao_FUNIL.xlsx', sheet_name='Vazao_FUNIL')
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 58030000 in result.columns
        assert 58060000 in result.columns
        
        # Check negative values were filtered
        assert (result[58030000] >= 0).all() or result[58030000].isna().any()
        assert (result[58060000] >= 0).all() or result[58060000].isna().any()
    
    @patch('flow_prediction_app.pd.read_csv')
    def test_load_meteorological_data_basic(self, mock_read_csv, sample_meteorological_data):
        """Test basic meteorological data loading."""
        mock_read_csv.return_value = sample_meteorological_data
        
        result = fpa.load_meteorological_data()
        
        # Check function was called correctly
        mock_read_csv.assert_called_once_with('../data/glob-funil-subbasin.csv')
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        
        # Check required columns are present
        predictor_cols = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
        for col in predictor_cols:
            assert col in result.columns
        
        assert 'year_x' in result.columns
        assert 'month_x' in result.columns
        assert 'ID_Subbasin' in result.columns
    
    def test_merge_data_basic(self, sample_monthly_flow_data, sample_meteorological_data):
        """Test data merging functionality."""
        result = fpa.merge_data(sample_monthly_flow_data, sample_meteorological_data)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 58030000 in result or 58060000 in result
        
        # Check merged data structure for available stations
        for flow_col, merged_df in result.items():
            assert isinstance(merged_df, pd.DataFrame)
            assert flow_col in merged_df.columns
            assert 'year_x' in merged_df.columns
            assert 'month_x' in merged_df.columns
            
            # Check predictor columns are present
            predictor_cols = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
            for col in predictor_cols:
                assert col in merged_df.columns
    
    def test_merge_data_empty_input(self):
        """Test merge_data with empty inputs."""
        empty_flow = pd.DataFrame()
        empty_met = pd.DataFrame()
        
        result = fpa.merge_data(empty_flow, empty_met)
        
        assert isinstance(result, dict)
        # Should return empty dict or dict with empty DataFrames

class TestModelTraining:
    """Test model training functionality."""
    
    def test_train_models_basic(self, sample_merged_data):
        """Test basic model training functionality."""
        # Ensure we have enough data for train/test split
        sample_merged_data['year_x'] = np.random.choice([2010, 2011, 2012, 2016, 2017], len(sample_merged_data))
        
        result = fpa.train_models(sample_merged_data, 58030000, train_year_cutoff=2015)
        
        if result is not None:  # Only test if we have sufficient data
            results, predictions = result
            
            # Check results structure
            assert isinstance(results, dict)
            assert isinstance(predictions, dict)
            
            # Check that we have results for multiple models
            expected_models = ['Linear_Regression', 'Ridge', 'Random_Forest', 'Gradient_Boosting', 'SVR']
            for model in expected_models:
                if model in results:
                    assert 'train' in results[model]
                    assert 'test' in results[model]
                    
                    # Check metrics structure
                    for dataset in ['train', 'test']:
                        metrics = results[model][dataset]
                        required_metrics = ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
                        for metric in required_metrics:
                            assert metric in metrics
    
    def test_train_models_insufficient_data(self):
        """Test train_models with insufficient data."""
        # Create dataset with all data in training period
        small_data = pd.DataFrame({
            'year_x': [2010, 2011, 2012],
            'month_x': [1, 2, 3],
            'u2_y': [2.0, 2.1, 2.2],
            'tmin_y': [20, 21, 22],
            'tmax_y': [30, 31, 32],
            'rs_y': [20, 21, 22],
            'rh_y': [75, 76, 77],
            'eto_y': [5, 5.1, 5.2],
            'pr_y': [100, 110, 120],
            58030000: [10, 11, 12]
        })
        
        result = fpa.train_models(small_data, 58030000, train_year_cutoff=2015)
        
        # Should return None for insufficient data
        assert result is None
    
    def test_train_models_edge_cases(self):
        """Test train_models with edge cases."""
        # Empty dataset
        empty_data = pd.DataFrame()
        result = fpa.train_models(empty_data, 58030000)
        assert result is None
        
        # Dataset with NaN values
        nan_data = pd.DataFrame({
            'year_x': [2010, 2016],
            'month_x': [1, 1],
            'u2_y': [np.nan, 2.0],
            'tmin_y': [20, np.nan],
            'tmax_y': [30, 31],
            'rs_y': [20, 21],
            'rh_y': [75, 76],
            'eto_y': [5, 5.1],
            'pr_y': [100, 110],
            58030000: [10, 11]
        })
        
        # Should handle NaN values gracefully
        result = fpa.train_models(nan_data, 58030000)
        # May return None or handle NaN appropriately

class TestMainFunction:
    """Test main application function."""
    
    @patch('flow_prediction_app.train_models')
    @patch('flow_prediction_app.merge_data')
    @patch('flow_prediction_app.load_meteorological_data')
    @patch('flow_prediction_app.process_flow_data')
    def test_main_function_flow(self, mock_process_flow, mock_load_met, mock_merge, mock_train):
        """Test main function execution flow."""
        # Setup mocks
        mock_process_flow.return_value = pd.DataFrame({'test': [1, 2, 3]})
        mock_load_met.return_value = pd.DataFrame({'test': [1, 2, 3]})
        mock_merge.return_value = {58030000: pd.DataFrame(), 58060000: pd.DataFrame()}
        mock_train.return_value = ({'model': {'train': {}, 'test': {}}}, {'model': {}})
        
        # Execute main function
        result = fpa.main()
        
        # Check that all functions were called
        mock_process_flow.assert_called_once()
        mock_load_met.assert_called_once()
        mock_merge.assert_called_once()
        
        # Check return structure
        assert isinstance(result, tuple)
        assert len(result) == 3  # results, predictions, datasets

class TestIntegrationScenarios:
    """Test integration scenarios with realistic data."""
    
    def test_complete_pipeline_small_dataset(self):
        """Test complete pipeline with small realistic dataset."""
        from tests.fixtures.test_data_generator import TestDataGenerator
        
        generator = TestDataGenerator()
        small_data = generator.generate_small_dataset(100)
        
        # Ensure proper train/test split
        small_data.loc[:60, 'year_x'] = 2014  # Training data
        small_data.loc[60:, 'year_x'] = 2017  # Test data
        
        # Test individual functions
        predictor_cols = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
        
        # Check data has required columns
        for col in predictor_cols:
            assert col in small_data.columns
        
        # Test metrics calculation with sample data
        observed = small_data[58030000][:10].values
        predicted = observed + np.random.normal(0, 0.1, len(observed))
        
        metrics = fpa.calculate_metrics(observed, predicted)
        assert all(metric in metrics for metric in ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe'])
        
        # Test model training if we have sufficient data
        if len(small_data) > 20:
            result = fpa.train_models(small_data, 58030000, train_year_cutoff=2015)
            if result is not None:
                results, predictions = result
                assert isinstance(results, dict)
                assert isinstance(predictions, dict)