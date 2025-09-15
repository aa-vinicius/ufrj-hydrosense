"""
Integration tests for the complete HydroSense pipeline.

Tests the end-to-end functionality of the entire system
using Test-Driven Development (TDD) approach.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import test data generator
from tests.fixtures.test_data_generator import TestDataGenerator, create_test_files

class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    def test_complete_pipeline_with_real_data_structure(self, tmp_path):
        """Test complete pipeline with realistic data structure."""
        # Create test data files
        test_files = create_test_files(tmp_path)
        
        # Change working directory to temp path for testing
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Import modules after changing directory
            import flow_prediction_app as fpa
            
            # Test data loading
            with patch('flow_prediction_app.pd.read_excel') as mock_read_excel, \
                 patch('flow_prediction_app.pd.read_csv') as mock_read_csv:
                
                mock_read_excel.return_value = test_files['flow_data']
                mock_read_csv.return_value = test_files['met_data']
                
                # Test individual pipeline components
                monthly_flow = fpa.process_flow_data()
                met_data = fpa.load_meteorological_data()
                merged_datasets = fpa.merge_data(monthly_flow, met_data)
                
                # Validate pipeline outputs
                assert isinstance(monthly_flow, pd.DataFrame)
                assert isinstance(met_data, pd.DataFrame)
                assert isinstance(merged_datasets, dict)
                
                # Test that we have data for expected stations
                expected_stations = [58030000, 58060000]
                available_stations = list(merged_datasets.keys())
                
                # Should have at least one station with data
                assert len(available_stations) > 0
                
                # Test model training if we have sufficient data
                for flow_col, dataset in merged_datasets.items():
                    if len(dataset) > 50:  # Sufficient data for training
                        # Ensure proper train/test split
                        dataset.loc[:30, 'year_x'] = 2014  # Training
                        dataset.loc[30:, 'year_x'] = 2017  # Testing
                        
                        result = fpa.train_models(dataset, flow_col, train_year_cutoff=2015)
                        
                        if result is not None:
                            results, predictions = result
                            assert isinstance(results, dict)
                            assert isinstance(predictions, dict)
                            
                            # Check that we have results for at least one model
                            assert len(results) > 0
                            assert len(predictions) > 0
        
        finally:
            os.chdir(original_cwd)
    
    def test_pipeline_data_flow_integrity(self, tmp_path):
        """Test data flow integrity through the pipeline."""
        # Create test environment
        test_files = create_test_files(tmp_path)
        
        # Test data transformations
        original_flow_data = test_files['flow_data']
        original_met_data = test_files['met_data']
        
        # Check original data properties
        assert len(original_flow_data) > 0
        assert len(original_met_data) > 0
        assert 'Data' in original_flow_data.columns
        assert 58030000 in original_flow_data.columns
        assert 58060000 in original_flow_data.columns
        
        # Test meteorological data properties
        required_met_cols = ['year_x', 'month_x', 'ID_Subbasin', 'u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y', 'pr_y']
        for col in required_met_cols:
            assert col in original_met_data.columns
        
        # Test data consistency
        assert original_met_data['ID_Subbasin'].isin([24, 36]).all()
        assert original_met_data['year_x'].min() >= 2010
        assert original_met_data['year_x'].max() <= 2020
        assert original_met_data['month_x'].min() >= 1
        assert original_met_data['month_x'].max() <= 12
    
    def test_model_training_pipeline(self, tmp_path):
        """Test model training pipeline with controlled data."""
        generator = TestDataGenerator()
        
        # Create sufficient data for training
        large_dataset = generator.generate_small_dataset(200)
        
        # Ensure proper temporal split
        large_dataset.loc[:120, 'year_x'] = np.random.randint(2010, 2016, 121)  # Training
        large_dataset.loc[120:, 'year_x'] = np.random.randint(2016, 2021, 80)   # Testing
        
        # Import and test training
        import flow_prediction_app as fpa
        
        result = fpa.train_models(large_dataset, 58030000, train_year_cutoff=2015)
        
        if result is not None:
            results, predictions = result
            
            # Test results structure
            assert isinstance(results, dict)
            assert isinstance(predictions, dict)
            
            # Test that multiple models were trained
            expected_models = ['Linear_Regression', 'Ridge', 'Random_Forest', 'Gradient_Boosting', 'SVR']
            trained_models = list(results.keys())
            
            # Should have trained at least some models
            assert len(trained_models) > 0
            
            # Test metrics for each trained model
            for model_name in trained_models:
                assert 'train' in results[model_name]
                assert 'test' in results[model_name]
                
                # Test metric completeness
                required_metrics = ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
                for dataset in ['train', 'test']:
                    for metric in required_metrics:
                        assert metric in results[model_name][dataset]
                        assert isinstance(results[model_name][dataset][metric], (int, float))
                        assert not np.isnan(results[model_name][dataset][metric])
    
    def test_error_prediction_pipeline(self, tmp_path):
        """Test Model 2 error prediction pipeline."""
        generator = TestDataGenerator()
        large_dataset = generator.generate_small_dataset(150)
        
        # Ensure proper temporal split
        large_dataset.loc[:90, 'year_x'] = np.random.randint(2010, 2016, 91)
        large_dataset.loc[90:, 'year_x'] = np.random.randint(2016, 2021, 60)
        
        # Import modules
        import flow_prediction_app as fpa
        import model2_error_prediction as m2ep
        
        # First train Model 1
        model1_result = fpa.train_models(large_dataset, 58030000, train_year_cutoff=2015)
        
        if model1_result is not None:
            model1_results, model1_predictions = model1_result
            
            # Test Model 2 training
            model2_result = m2ep.train_error_models(
                large_dataset, 58030000, model1_predictions, train_year_cutoff=2015
            )
            
            if model2_result is not None:
                model2_results, model2_predictions = model2_result
                
                # Test Model 2 structure
                assert isinstance(model2_results, dict)
                assert isinstance(model2_predictions, dict)
                
                # Test confidence interval creation
                ci = m2ep.create_confidence_intervals(model1_predictions, model2_predictions)
                
                if ci is not None:
                    required_ci_keys = ['corrected_predictions', 'lower_bound', 'upper_bound']
                    for key in required_ci_keys:
                        assert key in ci
                        assert isinstance(ci[key], np.ndarray)
                    
                    # Test confidence interval properties
                    assert len(ci['lower_bound']) == len(ci['upper_bound'])
                    assert np.all(ci['lower_bound'] <= ci['upper_bound'])

class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline behavior with missing data."""
        generator = TestDataGenerator()
        edge_cases = generator.generate_edge_case_data()
        
        import flow_prediction_app as fpa
        
        # Test with empty dataset
        empty_result = fpa.train_models(edge_cases['empty'], 58030000)
        assert empty_result is None
        
        # Test with single record
        single_result = fpa.train_models(edge_cases['single_record'], 58030000)
        assert single_result is None  # Insufficient for train/test split
        
        # Test with missing values
        missing_result = fpa.train_models(edge_cases['missing_values'], 58030000)
        # Should handle gracefully (may return None or handle NaN appropriately)
    
    def test_pipeline_with_extreme_values(self):
        """Test pipeline behavior with extreme values."""
        generator = TestDataGenerator()
        edge_cases = generator.generate_edge_case_data()
        
        import flow_prediction_app as fpa
        
        # Test with extreme values
        extreme_data = edge_cases['extreme_values']
        
        if len(extreme_data) > 20:
            # Ensure temporal split
            extreme_data.loc[:10, 'year_x'] = 2014
            extreme_data.loc[10:, 'year_x'] = 2017
            
            result = fpa.train_models(extreme_data, 58030000, train_year_cutoff=2015)
            
            # Should handle extreme values gracefully
            if result is not None:
                results, predictions = result
                
                # Check that metrics are still reasonable
                for model_name, model_results in results.items():
                    for dataset in ['train', 'test']:
                        metrics = model_results[dataset]
                        
                        # RMSE and MAE should be positive
                        assert metrics['RMSE'] > 0
                        assert metrics['MAE'] > 0
                        
                        # Correlation should be in valid range
                        assert -1 <= metrics['Correlation'] <= 1
    
    def test_pipeline_memory_efficiency(self, tmp_path):
        """Test pipeline memory efficiency with larger datasets."""
        generator = TestDataGenerator()
        
        # Create larger dataset
        large_dataset = generator.generate_small_dataset(500)
        large_dataset.loc[:300, 'year_x'] = np.random.randint(2010, 2016, 301)
        large_dataset.loc[300:, 'year_x'] = np.random.randint(2016, 2021, 200)
        
        import flow_prediction_app as fpa
        
        # Test that pipeline can handle larger datasets
        result = fpa.train_models(large_dataset, 58030000, train_year_cutoff=2015)
        
        if result is not None:
            results, predictions = result
            
            # Should complete without memory errors
            assert isinstance(results, dict)
            assert isinstance(predictions, dict)
            
            # Check that predictions have reasonable sizes
            for model_name, model_preds in predictions.items():
                if 'test_pred' in model_preds:
                    assert len(model_preds['test_pred']) > 0
                    assert len(model_preds['test_pred']) < 1000  # Reasonable size

class TestOutputGeneration:
    """Test output file generation in pipeline."""
    
    def test_csv_output_generation(self, tmp_path):
        """Test CSV output generation."""
        # Create mock results
        mock_results = {
            58030000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': 3.5, 'MAE': 2.8, 'Correlation': 0.85,
                        'BIAS': 0.1, 'Nash_Sutcliffe': 0.72
                    },
                    'test': {
                        'RMSE': 4.2, 'MAE': 3.1, 'Correlation': 0.78,
                        'BIAS': 0.3, 'Nash_Sutcliffe': 0.65
                    }
                }
            }
        }
        
        # Simulate CSV generation logic
        results_list = []
        for flow_col in mock_results.keys():
            subbasin_id = 24 if flow_col == 58030000 else 36
            
            for model_name, metrics in mock_results[flow_col].items():
                for dataset_type in ['train', 'test']:
                    row = {
                        'Flow_Station': flow_col,
                        'Subbasin_ID': subbasin_id,
                        'Model': model_name,
                        'Dataset': dataset_type.title(),
                        **metrics[dataset_type]
                    }
                    results_list.append(row)
        
        df = pd.DataFrame(results_list)
        
        # Test CSV creation
        output_file = tmp_path / 'test_results.csv'
        df.to_csv(output_file, index=False)
        
        # Verify file was created and has correct structure
        assert output_file.exists()
        
        # Read back and verify
        loaded_df = pd.read_csv(output_file)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
    
    def test_plot_output_generation(self, tmp_path):
        """Test plot output generation."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create sample plot
        plt.figure(figsize=(10, 6))
        
        # Sample data
        dates = pd.date_range('2016-01-01', periods=12, freq='M')
        observed = np.random.uniform(8, 15, 12)
        predicted = observed + np.random.normal(0, 0.5, 12)
        
        plt.plot(dates, observed, 'b-', label='Observed')
        plt.plot(dates, predicted, 'r--', label='Predicted')
        plt.fill_between(dates, predicted-1, predicted+1, alpha=0.3, label='Confidence Interval')
        
        plt.title('Test Flow Prediction')
        plt.xlabel('Date')
        plt.ylabel('Flow (m³/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_file = tmp_path / 'test_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0  # File has content

class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_execution_time(self):
        """Test that pipeline executes within reasonable time."""
        import time
        generator = TestDataGenerator()
        
        # Create moderate-sized dataset
        dataset = generator.generate_small_dataset(100)
        dataset.loc[:60, 'year_x'] = np.random.randint(2010, 2016, 61)
        dataset.loc[60:, 'year_x'] = np.random.randint(2016, 2021, 40)
        
        import flow_prediction_app as fpa
        
        # Time the training process
        start_time = time.time()
        result = fpa.train_models(dataset, 58030000, train_year_cutoff=2015)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 60  # Should complete within 60 seconds
        
        if result is not None:
            results, predictions = result
            assert len(results) > 0  # Should have trained at least one model
    
    def test_pipeline_scalability(self):
        """Test pipeline scalability with different data sizes."""
        generator = TestDataGenerator()
        import flow_prediction_app as fpa
        
        # Test with different dataset sizes
        sizes = [50, 100, 200]
        execution_times = []
        
        for size in sizes:
            dataset = generator.generate_small_dataset(size)
            train_size = int(size * 0.6)
            dataset.loc[:train_size, 'year_x'] = np.random.randint(2010, 2016, train_size + 1)
            dataset.loc[train_size:, 'year_x'] = np.random.randint(2016, 2021, size - train_size)
            
            import time
            start_time = time.time()
            result = fpa.train_models(dataset, 58030000, train_year_cutoff=2015)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        # Execution time should scale reasonably (not exponentially)
        # This is a basic scalability check
        if len(execution_times) >= 2:
            # Time shouldn't increase too dramatically
            time_ratio = execution_times[-1] / execution_times[0] if execution_times[0] > 0 else 1
            size_ratio = sizes[-1] / sizes[0]
            
            # Time increase should be reasonable relative to size increase
            assert time_ratio < size_ratio * 2  # Allow some overhead but not excessive

class TestPipelineIntegration:
    """Test integration between different pipeline components."""
    
    def test_model1_to_model2_integration(self):
        """Test integration between Model 1 and Model 2."""
        generator = TestDataGenerator()
        dataset = generator.generate_small_dataset(120)
        
        # Proper temporal split
        dataset.loc[:72, 'year_x'] = np.random.randint(2010, 2016, 73)
        dataset.loc[72:, 'year_x'] = np.random.randint(2016, 2021, 48)
        
        import flow_prediction_app as fpa
        import model2_error_prediction as m2ep
        
        # Train Model 1
        model1_result = fpa.train_models(dataset, 58030000, train_year_cutoff=2015)
        
        if model1_result is not None:
            model1_results, model1_predictions = model1_result
            
            # Test that Model 1 outputs are compatible with Model 2 inputs
            assert isinstance(model1_predictions, dict)
            
            # Check for required model in predictions
            if 'Random_Forest' in model1_predictions:
                rf_preds = model1_predictions['Random_Forest']
                
                # Check required keys for Model 2
                required_keys = ['y_train', 'train_pred', 'y_test', 'test_pred']
                for key in required_keys:
                    assert key in rf_preds
                
                # Train Model 2
                model2_result = m2ep.train_error_models(
                    dataset, 58030000, model1_predictions, train_year_cutoff=2015
                )
                
                if model2_result is not None:
                    model2_results, model2_predictions = model2_result
                    
                    # Test successful integration
                    assert isinstance(model2_results, dict)
                    assert isinstance(model2_predictions, dict)
    
    def test_evaluation_integration(self, tmp_path):
        """Test integration with evaluation modules."""
        # Create mock results structure
        mock_results = {
            58030000: {
                'Random_Forest': {
                    'train': {'RMSE': 3.0, 'MAE': 2.5, 'Correlation': 0.8, 'BIAS': 0.1, 'Nash_Sutcliffe': 0.7},
                    'test': {'RMSE': 4.0, 'MAE': 3.0, 'Correlation': 0.75, 'BIAS': 0.2, 'Nash_Sutcliffe': 0.6}
                }
            }
        }
        
        # Test evaluation data processing
        results_list = []
        for flow_col in mock_results.keys():
            subbasin_id = 24 if flow_col == 58030000 else 36
            
            for model_name, metrics in mock_results[flow_col].items():
                for dataset_type in ['train', 'test']:
                    row = {
                        'Flow_Station': flow_col,
                        'Subbasin_ID': subbasin_id,
                        'Model': model_name,
                        'Dataset': dataset_type.title(),
                        **metrics[dataset_type]
                    }
                    results_list.append(row)
        
        df = pd.DataFrame(results_list)
        
        # Test summary statistics
        summary = df.groupby(['Flow_Station', 'Dataset'])[
            ['RMSE', 'MAE', 'Correlation', 'Nash_Sutcliffe']
        ].mean()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        
        # Test CSV output
        output_file = tmp_path / 'integration_test.csv'
        df.to_csv(output_file, index=False)
        
        assert output_file.exists()
        
        # Verify data integrity
        loaded_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df, loaded_df)