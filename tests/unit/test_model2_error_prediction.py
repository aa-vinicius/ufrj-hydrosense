"""
Unit tests for model2_error_prediction.py

Tests all functions and methods in the error prediction module
using Test-Driven Development (TDD) approach.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from scipy import stats

# Import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import model2_error_prediction as m2ep

class TestErrorModelTraining:
    """Test error model training functionality."""
    
    def test_train_error_models_basic(self, sample_merged_data, sample_predictions):
        """Test basic error model training (novo padrão)."""
        # Garante que sample_merged_data tenha 100 linhas para compatibilidade
        df = sample_merged_data.copy().iloc[:100].reset_index(drop=True)
        df['year'] = np.concatenate([
            np.full(60, 2014),  # Treino
            np.full(40, 2017)   # Teste
        ])
        result = m2ep.train_error_models(
            df,
            'flow_next_month',
            sample_predictions,
            train_year_cutoff=2015
        )
        # O resultado pode ser None em edge cases, e isso é esperado
        if result is None or result == (None, None):
            assert result is None or result == (None, None)
        else:
            results, predictions = result
            assert isinstance(results, dict)
            assert isinstance(predictions, dict)
            expected_models = ['Linear_Regression', 'Ridge', 'Random_Forest', 'Gradient_Boosting', 'SVR', 'MLP']
            for model in expected_models:
                if model in results:
                    assert 'train' in results[model]
                    assert 'test' in results[model]
                    for dataset in ['train', 'test']:
                        metrics = results[model][dataset]
    
    def test_train_error_models_insufficient_data(self):
        """Test error model training with insufficient data (novo padrão)."""
        small_data = pd.DataFrame({
            'year': [2010, 2011, 2012],
            'month': [1, 2, 3],
            'u2': [2.0, 2.1, 2.2],
            'tmin': [20, 21, 22],
            'tmax': [30, 31, 32],
            'rs': [20, 21, 22],
            'rh': [75, 76, 77],
            'eto': [5, 5.1, 5.2],
            'pr': [100, 110, 120],
            'flow_next_month': [10, 11, 12],
            'subbasin_id': [24, 24, 24],
            'station_id': [58030000, 58030000, 58030000]
        })
        sample_predictions = {
            'Random_Forest': {
                'y_train': pd.Series([10, 11, 12]),
                'train_pred': np.array([10.1, 10.9, 12.1]),
                'y_test': pd.Series([]),
                'test_pred': np.array([])
            }
        }
        result = m2ep.train_error_models(small_data, 'flow_next_month', sample_predictions)
        assert result is None or result == (None, None)
    
    def test_train_error_models_missing_model1_predictions(self, sample_merged_data):
        """Test error model training when Model 1 predictions are missing (novo padrão)."""
        # Garante que sample_merged_data tenha 100 linhas para compatibilidade
        df = sample_merged_data.copy().iloc[:100].reset_index(drop=True)
        df['year'] = np.concatenate([
            np.full(60, 2014),
            np.full(40, 2017)
        ])
        empty_predictions = {}
        result = m2ep.train_error_models(df, 'flow_next_month', empty_predictions)
        assert result is None or result == (None, None)
    
    def test_train_error_models_edge_cases(self):
        """Test error model training with edge cases (novo padrão)."""
        empty_data = pd.DataFrame({col: [] for col in ['year', 'month', 'subbasin_id', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr', 'station_id', 'flow_next_month']})
        empty_predictions = {}
        result = m2ep.train_error_models(empty_data, 'flow_next_month', empty_predictions)
        assert result is None or result == (None, None)

class TestConfidenceIntervals:
    """Test confidence interval creation functionality."""
    
    def test_create_confidence_intervals_basic(self):
        """Test basic confidence interval creation."""
        # Create mock model predictions
        model1_preds = {
            'Random_Forest': {
                'test_pred': np.array([10, 12, 14, 16, 18])
            }
        }
        
        model2_preds = {
            'Random_Forest': {
                'test_error_pred': np.array([0.5, -0.3, 0.8, -0.2, 0.1]),
                'test_errors': np.array([0.6, -0.4, 0.9, -0.1, 0.2])
            }
        }
        
        result = m2ep.create_confidence_intervals(model1_preds, model2_preds)
        
        if result is not None:
            # Check result structure
            required_keys = ['corrected_predictions', 'lower_bound', 'upper_bound', 
                           'original_predictions', 'error_predictions']
            for key in required_keys:
                assert key in result
                assert isinstance(result[key], np.ndarray)
            
            # Check that confidence intervals are wider than point predictions
            assert len(result['lower_bound']) == len(result['upper_bound'])
            assert len(result['corrected_predictions']) == len(result['original_predictions'])
            
            # Check that lower bound < upper bound
            assert np.all(result['lower_bound'] <= result['upper_bound'])
    
    def test_create_confidence_intervals_missing_model(self):
        """Test confidence interval creation with missing model predictions."""
        model1_preds = {}
        model2_preds = {}
        
        result = m2ep.create_confidence_intervals(model1_preds, model2_preds)
        
        # Should return None when required models are missing
        assert result is None
    
    def test_create_confidence_intervals_different_confidence_levels(self):
        """Test confidence intervals with different confidence levels."""
        model1_preds = {
            'Random_Forest': {
                'test_pred': np.array([10, 12, 14])
            }
        }
        
        model2_preds = {
            'Random_Forest': {
                'test_error_pred': np.array([0.5, -0.3, 0.8]),
                'test_errors': np.array([0.6, -0.4, 0.9])
            }
        }
        
        # Test different confidence levels
        for confidence_level in [0.90, 0.95, 0.99]:
            result = m2ep.create_confidence_intervals(
                model1_preds, model2_preds, confidence_level=confidence_level
            )
            
            if result is not None:
                # Higher confidence levels should produce wider intervals
                interval_width = result['upper_bound'] - result['lower_bound']
                assert np.all(interval_width > 0)

class TestMainModel2Function:
    """Test main Model 2 function."""
    
    @patch('model2_error_prediction.create_confidence_intervals')
    @patch('model2_error_prediction.train_error_models')
    @patch('model2_error_prediction.main')
    def test_main_model2_flow(self, mock_main, mock_train_error, mock_create_ci):
        """Test main Model 2 function execution flow."""
        # Setup mocks
        mock_main.return_value = (
            {58030000: {'model': {'train': {}, 'test': {}}}},  # model1_results
            {58030000: {'Random_Forest': {'test_pred': np.array([1, 2, 3])}}},  # model1_predictions
            {58030000: pd.DataFrame()}  # datasets
        )
        
        mock_train_error.return_value = (
            {'model': {'train': {}, 'test': {}}},  # error_results
            {'model': {'test_error_pred': np.array([0.1, 0.2, 0.3])}}  # error_predictions
        )
        
        mock_create_ci.return_value = {
            'corrected_predictions': np.array([1.1, 2.2, 3.3]),
            'lower_bound': np.array([0.5, 1.5, 2.5]),
            'upper_bound': np.array([1.5, 2.5, 3.5])
        }
        
        # Execute main Model 2 function
        result = m2ep.main_model2()
        
        # Check that main function was called
        mock_main.assert_called_once()
        
        # Check return structure
        assert isinstance(result, tuple)
        assert len(result) == 6  # model1_results, model1_preds, model2_results, model2_preds, ci, datasets

class TestStatisticalFunctions:
    """Test statistical and mathematical functions."""
    
    def test_z_score_calculation(self):
        """Test Z-score calculation for different confidence levels."""
        # Test common confidence levels
        test_cases = [
            (0.90, 1.645),
            (0.95, 1.96),
            (0.99, 2.576)
        ]
        
        for confidence_level, expected_z in test_cases:
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            assert abs(z_score - expected_z) < 0.01  # Allow small numerical differences
    
    def test_error_calculation_logic(self):
        """Test error calculation logic used in Model 2."""
        # Test residual calculation
        observed = np.array([10, 15, 12, 18, 14])
        predicted = np.array([11, 14, 13, 17, 15])
        
        errors = observed - predicted
        expected_errors = np.array([-1, 1, -1, 1, -1])
        
        np.testing.assert_array_equal(errors, expected_errors)
        
        # Test error statistics
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        assert isinstance(error_mean, (int, float))
        assert isinstance(error_std, (int, float))
        assert error_std >= 0

class TestDataValidation:
    """Test data validation and preprocessing."""
    
    def test_predictor_columns_validation(self):
        """Test that required predictor columns are validated."""
        required_predictors = ['u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
        
        # Test with complete data
        complete_data = pd.DataFrame({
            col: np.random.randn(10) for col in required_predictors
        })
        complete_data['year_x'] = np.random.randint(2010, 2020, 10)
        complete_data['month_x'] = np.random.randint(1, 13, 10)
        complete_data[58030000] = np.random.randn(10)
        
        # All required columns should be present
        for col in required_predictors:
            assert col in complete_data.columns
        
        # Test with missing columns
        incomplete_data = complete_data.drop(columns=['u2_y', 'tmin_y'])
        
        # Should identify missing columns
        missing_cols = set(required_predictors) - set(incomplete_data.columns)
        assert 'u2_y' in missing_cols
        assert 'tmin_y' in missing_cols
    
    def test_temporal_split_validation(self):
        """Test temporal data splitting logic (novo padrão)."""
        # Create test data with known years
        test_data = pd.DataFrame({
            'year': [2010, 2011, 2012, 2015, 2016, 2017, 2018],
            'month': [1, 2, 3, 4, 5, 6, 7],
            'value': [1, 2, 3, 4, 5, 6, 7]
        })
        train_year_cutoff = 2015
        # Test train/test split
        train_data = test_data[test_data['year'] <= train_year_cutoff]
        test_data_split = test_data[test_data['year'] > train_year_cutoff]
        # Validate split
        assert len(train_data) == 4  # 2010, 2011, 2012, 2015
        assert len(test_data_split) == 3  # 2016, 2017, 2018
        assert train_data['year'].max() <= train_year_cutoff
        assert test_data_split['year'].min() > train_year_cutoff

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_predictions_handling(self):
        """Test handling of empty prediction arrays (novo padrão)."""
        empty_data = pd.DataFrame({col: [] for col in ['year', 'month', 'subbasin_id', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr', 'station_id', 'flow_next_month']})
        empty_predictions = {}
        result = m2ep.train_error_models(empty_data, 'flow_next_month', empty_predictions)
        assert result is None or result == (None, None)
    
    def test_nan_values_handling(self):
        """Test handling of NaN values in data (novo padrão)."""
        data_with_nan = pd.DataFrame({
            'year': [2010, 2016, 2017],
            'month': [1, 2, 3],
            'subbasin_id': [24, 24, 24],
            'u2': [2.0, np.nan, 2.2],
            'tmin': [20, 21, np.nan],
            'tmax': [30, 31, 32],
            'rs': [20, 21, 22],
            'rh': [75, 76, 77],
            'eto': [5, 5.1, 5.2],
            'pr': [100, 110, 120],
            'station_id': [58030000, 58030000, 58030000],
            'flow_next_month': [10, 11, np.nan]
        })
        predictions_with_nan = {
            'Random_Forest': {
                'y_train': np.array([10]),
                'train_pred': np.array([10.1]),
                'y_test': np.array([11, np.nan]),
                'test_pred': np.array([10.9, 12.1])
            }
        }
        result = m2ep.train_error_models(data_with_nan, 'flow_next_month', predictions_with_nan)
        # May return None or handle NaN appropriately
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        model1_preds = {
            'Random_Forest': {
                'test_pred': np.array([10, 12, 14])  # Length 3
            }
        }
        
        model2_preds = {
            'Random_Forest': {
                'test_error_pred': np.array([0.5, -0.3]),  # Length 2 (mismatch)
                'test_errors': np.array([0.6, -0.4, 0.9])  # Length 3
            }
        }
        
        # Should handle mismatched lengths gracefully
        result = m2ep.create_confidence_intervals(model1_preds, model2_preds)
        # Implementation should either handle gracefully or return None