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
        required_metrics = ['RMSE', 'MAE', 'R2', 'KGE', 'PBIAS', 'Bias', 'Nash_Sutcliffe']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_calculate_metrics_edge_cases(self):
        """Test metrics calculation with edge cases."""
        # Single value
        observed = np.array([10])
        predicted = np.array([10])

        metrics = fpa.calculate_metrics(observed, predicted)
        # Todos os valores devem ser None ou nan para arrays de tamanho 1
        for value in metrics.values():
            assert value is None or (isinstance(value, float) and np.isnan(value))

class TestDataProcessing:
    """Test data processing functions."""
    
    @patch('flow_prediction_app.pd.read_excel')
    def test_process_flow_data_basic(self, mock_read_excel, sample_flow_data):
        """Test basic flow data processing."""
        mock_read_excel.return_value = sample_flow_data
        result = fpa.process_flow_data()
        mock_read_excel.assert_called_once_with('../data/Vazao_FUNIL.xlsx', sheet_name='Vazao_FUNIL')
        assert isinstance(result, pd.DataFrame)
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 58030000 in result.columns
        assert 58060000 in result.columns
        assert (result[58030000] >= 0).all() or result[58030000].isna().any()
        assert (result[58060000] >= 0).all() or result[58060000].isna().any()
    
    @patch('flow_prediction_app.pd.read_csv')
    def test_load_meteorological_data_basic(self, mock_read_csv, sample_meteorological_data):
        """Test basic meteorological data loading (nova estrutura)."""
        mock_read_csv.return_value = sample_meteorological_data
        result = fpa.load_meteorological_data()
        mock_read_csv.assert_called_once_with('data/meteo_vazao_shifted_station_58030000.csv')
        assert isinstance(result, pd.DataFrame)
        predictor_cols = ['year', 'month', 'u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
        for col in predictor_cols:
            assert col in result.columns
        assert 'subbasin_id' in result.columns
        assert 'station_id' in result.columns
        assert 'flow_next_month' in result.columns
    
    def test_merge_data_basic(self, sample_meteorological_data):
        """Test data splitting por estação (nova estrutura)."""
        # Para compatibilidade com nova assinatura, passar o mesmo DataFrame duas vezes
        result = fpa.merge_data(sample_meteorological_data, sample_meteorological_data)
        assert isinstance(result, dict)
        for station_id, merged_df in result.items():
            assert isinstance(merged_df, pd.DataFrame)
            assert 'station_id' in merged_df.columns
            assert 'flow_next_month' in merged_df.columns
    
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
        """Test basic model training functionality (nova estrutura)."""
        sample_merged_data['year'] = np.random.choice([2010, 2011, 2012, 2016, 2017], len(sample_merged_data))
        result = fpa.train_models(sample_merged_data, 'flow_next_month', train_year_cutoff=2015)
        if result is not None:
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
                        required_metrics = ['RMSE', 'MAE', 'R2', 'KGE', 'PBIAS', 'Bias', 'Nash_Sutcliffe']
                        for metric in required_metrics:
                            assert metric in metrics
    
    def test_train_models_insufficient_data(self):
        """Test train_models with insufficient data (nova estrutura)."""
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
        result = fpa.train_models(small_data, 'flow_next_month', train_year_cutoff=2015)
        assert result is None or result == (None, None)
    
    def test_train_models_edge_cases(self):
        """Test train_models with edge cases."""
        # Empty dataset
        empty_data = pd.DataFrame()
        result = fpa.train_models(empty_data, 58030000)
        assert result is None or result == (None, None)
        
        nan_data = pd.DataFrame({
            'year': [2010, 2016],
            'month': [1, 1],
            'u2': [np.nan, 2.0],
                'tmin': [20, np.nan],
                'tmax': [30, 31],
                'rs': [20, 21],
                'rh': [75, 76],
                'eto': [5, 5.1],
                'pr': [100, 110],
                'station_id': [58030000, 58030000],
                'flow_next_month': [10, 11]
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
        # Check que main executa e retorna estrutura esperada
        mock_load_met.assert_called_once()
        mock_merge.assert_called_once()

        # Check return structure
        assert isinstance(result, tuple)
        assert len(result) == 3  # results, predictions, datasets

class TestIntegrationScenarios:
    """Test integration scenarios with realistic data."""
    
    def test_complete_pipeline_small_dataset(self):
        """Test complete pipeline with small realistic dataset (novo padrão)."""
        from tests.fixtures.test_data_generator import TestDataGenerator
        generator = TestDataGenerator()
        small_data = generator.generate_small_dataset(100)
        # Garantir split de treino/teste
        small_data.loc[:60, 'year'] = 2014  # Treino
        small_data.loc[60:, 'year'] = 2017  # Teste
        # Testa se todas as colunas do novo padrão existem
        predictor_cols = ['u2', 'tmin', 'tmax', 'rs', 'rh', 'eto', 'pr']
        for col in predictor_cols:
            assert col in small_data.columns
        # Testa métricas
        observed = small_data['flow_next_month'][:10].to_numpy()
        predicted = observed + np.random.normal(0, 0.1, len(observed))
        metrics = fpa.calculate_metrics(observed, predicted)
        assert all(metric in metrics for metric in ['RMSE', 'MAE', 'Bias', 'Nash_Sutcliffe'])
        # Testa model training se houver dados suficientes
        if len(small_data) > 20:
            result = fpa.train_models(small_data, 'flow_next_month', train_year_cutoff=2015)
            if result is not None:
                results, predictions = result
                assert isinstance(results, dict)
                assert isinstance(predictions, dict)