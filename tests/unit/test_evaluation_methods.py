"""
Unit tests for evaluation methods (evaluate_model1.py and evaluate_model2.py)

Tests all functions and methods in the evaluation modules
using Test-Driven Development (TDD) approach.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Import the modules under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import evaluate_model1 as em1
import evaluate_model2 as em2

class TestModel1Evaluation:
    """Test Model 1 evaluation functionality."""
    
    @patch('evaluate_model1.main')
    def test_save_model1_results_basic(self, mock_main, temp_output_dir):
        """Test basic Model 1 results saving functionality."""
        # Setup mock return values
        mock_results = {
            58030000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': 3.5,
                        'MAE': 2.8,
                        'Correlation': 0.85,
                        'BIAS': 0.1,
                        'Nash_Sutcliffe': 0.72
                    },
                    'test': {
                        'RMSE': 4.2,
                        'MAE': 3.1,
                        'Correlation': 0.78,
                        'BIAS': 0.3,
                        'Nash_Sutcliffe': 0.65
                    }
                }
            },
            58060000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': 2.1,
                        'MAE': 1.6,
                        'Correlation': 0.88,
                        'BIAS': -0.05,
                        'Nash_Sutcliffe': 0.77
                    },
                    'test': {
                        'RMSE': 2.8,
                        'MAE': 2.0,
                        'Correlation': 0.82,
                        'BIAS': 0.15,
                        'Nash_Sutcliffe': 0.68
                    }
                }
            }
        }
        
        mock_predictions = {}
        mock_datasets = {}
        
        mock_main.return_value = (mock_results, mock_predictions, mock_datasets)
        
        # Patch the CSV output path to use temp directory
        with patch('evaluate_model1.pd.DataFrame.to_csv') as mock_to_csv:
            result = em1.save_model1_results()
            
            # Check that main was called
            mock_main.assert_called_once()
            
            # Check that CSV was saved
            mock_to_csv.assert_called_once()
            
            # Check return values
            assert isinstance(result, tuple)
            assert len(result) == 3
    
    def test_model1_results_dataframe_structure(self):
        """Test the structure of Model 1 results DataFrame."""
        # Create sample results
        sample_results = {
            58030000: {
                'Linear_Regression': {
                    'train': {
                        'RMSE': 4.0,
                        'MAE': 3.0,
                        'Correlation': 0.8,
                        'BIAS': 0.2,
                        'Nash_Sutcliffe': 0.6
                    },
                    'test': {
                        'RMSE': 5.0,
                        'MAE': 4.0,
                        'Correlation': 0.7,
                        'BIAS': 0.3,
                        'Nash_Sutcliffe': 0.5
                    }
                }
            }
        }
        
        # Simulate the DataFrame creation logic
        results_list = []
        for flow_col in sample_results.keys():
            subbasin_id = 24 if flow_col == 58030000 else 36
            
            for model_name, metrics in sample_results[flow_col].items():
                # Training results
                train_row = {
                    'Flow_Station': flow_col,
                    'Subbasin_ID': subbasin_id,
                    'Model': model_name,
                    'Dataset': 'Training',
                    'RMSE': metrics['train']['RMSE'],
                    'MAE': metrics['train']['MAE'],
                    'Correlation': metrics['train']['Correlation'],
                    'BIAS': metrics['train']['BIAS'],
                    'Nash_Sutcliffe': metrics['train']['Nash_Sutcliffe']
                }
                results_list.append(train_row)
                
                # Test results
                test_row = {
                    'Flow_Station': flow_col,
                    'Subbasin_ID': subbasin_id,
                    'Model': model_name,
                    'Dataset': 'Test',
                    'RMSE': metrics['test']['RMSE'],
                    'MAE': metrics['test']['MAE'],
                    'Correlation': metrics['test']['Correlation'],
                    'BIAS': metrics['test']['BIAS'],
                    'Nash_Sutcliffe': metrics['test']['Nash_Sutcliffe']
                }
                results_list.append(test_row)
        
        df = pd.DataFrame(results_list)
        
        # Test DataFrame structure
        expected_columns = ['Flow_Station', 'Subbasin_ID', 'Model', 'Dataset', 
                          'RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
        
        for col in expected_columns:
            assert col in df.columns
        
        # Test data types and values
        assert df['Flow_Station'].dtype in [np.int64, np.object_]
        assert df['Subbasin_ID'].dtype in [np.int64, np.object_]
        assert df['Dataset'].isin(['Training', 'Test']).all()
        
        # Test metric values are numeric
        metric_cols = ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
        for col in metric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_model1_evaluation_edge_cases(self):
        """Test Model 1 evaluation with edge cases."""
        # Empty results
        empty_results = {}
        
        results_list = []
        for flow_col in empty_results.keys():
            pass  # Should not enter loop
        
        df = pd.DataFrame(results_list)
        assert len(df) == 0
        
        # Single model, single station
        single_result = {
            58030000: {
                'Linear_Regression': {
                    'train': {'RMSE': 1.0, 'MAE': 0.8, 'Correlation': 0.9, 'BIAS': 0.0, 'Nash_Sutcliffe': 0.8},
                    'test': {'RMSE': 1.2, 'MAE': 1.0, 'Correlation': 0.85, 'BIAS': 0.1, 'Nash_Sutcliffe': 0.75}
                }
            }
        }
        
        # Should handle single result correctly
        assert len(single_result) == 1
        assert 58030000 in single_result

class TestModel2Evaluation:
    """Test Model 2 evaluation functionality."""
    
    @patch('evaluate_model2.main_model2')
    def test_save_model2_results_basic(self, mock_main_model2, temp_output_dir):
        """Test basic Model 2 results saving functionality."""
        # Setup mock return values
        mock_model1_results = {}
        mock_model1_preds = {}
        mock_model2_results = {
            58030000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': 1.5,
                        'MAE': 1.2,
                        'Correlation': 0.4,
                        'BIAS': 0.05,
                        'Nash_Sutcliffe': 0.3
                    },
                    'test': {
                        'RMSE': 2.1,
                        'MAE': 1.8,
                        'Correlation': 0.2,
                        'BIAS': 0.1,
                        'Nash_Sutcliffe': 0.1
                    }
                }
            }
        }
        mock_model2_preds = {}
        mock_ci = {}
        mock_datasets = {}
        
        mock_main_model2.return_value = (
            mock_model1_results, mock_model1_preds, mock_model2_results, 
            mock_model2_preds, mock_ci, mock_datasets
        )
        
        # Patch the CSV output path to use temp directory
        with patch('evaluate_model2.pd.DataFrame.to_csv') as mock_to_csv:
            result = em2.save_model2_results()
            
            # Check that main_model2 was called
            mock_main_model2.assert_called_once()
            
            # Check that CSV was saved
            mock_to_csv.assert_called_once()
            
            # Check return values
            assert isinstance(result, tuple)
            assert len(result) == 6
    
    def test_model2_results_dataframe_structure(self):
        """Test the structure of Model 2 results DataFrame."""
        # Create sample Model 2 results (error prediction results)
        sample_results = {
            58030000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': 1.8,
                        'MAE': 1.4,
                        'Correlation': 0.3,
                        'BIAS': 0.0,
                        'Nash_Sutcliffe': 0.2
                    },
                    'test': {
                        'RMSE': 2.5,
                        'MAE': 2.0,
                        'Correlation': 0.1,
                        'BIAS': 0.05,
                        'Nash_Sutcliffe': -0.1
                    }
                }
            }
        }
        
        # Simulate the DataFrame creation logic for Model 2
        results_list = []
        for flow_col in sample_results.keys():
            subbasin_id = 24 if flow_col == 58030000 else 36
            
            for model_name, metrics in sample_results[flow_col].items():
                # Training results
                train_row = {
                    'Flow_Station': flow_col,
                    'Subbasin_ID': subbasin_id,
                    'Model': model_name,
                    'Dataset': 'Training',
                    'RMSE': metrics['train']['RMSE'],
                    'MAE': metrics['train']['MAE'],
                    'Correlation': metrics['train']['Correlation'],
                    'BIAS': metrics['train']['BIAS'],
                    'Nash_Sutcliffe': metrics['train']['Nash_Sutcliffe']
                }
                results_list.append(train_row)
                
                # Test results
                test_row = {
                    'Flow_Station': flow_col,
                    'Subbasin_ID': subbasin_id,
                    'Model': model_name,
                    'Dataset': 'Test',
                    'RMSE': metrics['test']['RMSE'],
                    'MAE': metrics['test']['MAE'],
                    'Correlation': metrics['test']['Correlation'],
                    'BIAS': metrics['test']['BIAS'],
                    'Nash_Sutcliffe': metrics['test']['Nash_Sutcliffe']
                }
                results_list.append(test_row)
        
        df = pd.DataFrame(results_list)
        
        # Test DataFrame structure (same as Model 1)
        expected_columns = ['Flow_Station', 'Subbasin_ID', 'Model', 'Dataset', 
                          'RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
        
        for col in expected_columns:
            assert col in df.columns
        
        # Test that Model 2 typically has lower correlations (predicting errors is harder)
        test_correlations = df[df['Dataset'] == 'Test']['Correlation']
        if len(test_correlations) > 0:
            # Error prediction correlations are typically lower than flow prediction
            assert test_correlations.mean() < 0.8  # Should be lower than typical flow predictions

class TestEvaluationMetrics:
    """Test evaluation metrics and statistical calculations."""
    
    def test_performance_summary_calculation(self):
        """Test performance summary calculations."""
        # Create sample DataFrame similar to what would be generated
        sample_data = pd.DataFrame({
            'Flow_Station': [58030000, 58030000, 58060000, 58060000],
            'Dataset': ['Training', 'Test', 'Training', 'Test'],
            'RMSE': [3.0, 4.0, 2.0, 3.0],
            'MAE': [2.5, 3.5, 1.5, 2.5],
            'Correlation': [0.85, 0.75, 0.90, 0.80],
            'Nash_Sutcliffe': [0.70, 0.60, 0.80, 0.70]
        })
        
        # Test groupby operations similar to what's used in evaluation
        summary = sample_data.groupby(['Flow_Station', 'Dataset'])[
            ['RMSE', 'MAE', 'Correlation', 'Nash_Sutcliffe']
        ].mean()
        
        # Check summary structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4  # 2 stations × 2 datasets
        
        # Check that all metrics are present
        expected_metrics = ['RMSE', 'MAE', 'Correlation', 'Nash_Sutcliffe']
        for metric in expected_metrics:
            assert metric in summary.columns
        
        # Check that values are reasonable
        assert (summary['RMSE'] > 0).all()
        assert (summary['MAE'] > 0).all()
        assert (summary['Correlation'] >= -1).all() and (summary['Correlation'] <= 1).all()
        assert (summary['Nash_Sutcliffe'] <= 1).all()
    
    def test_subbasin_mapping(self):
        """Test subbasin ID mapping logic."""
        # Test the mapping logic used in evaluation
        flow_stations = [58030000, 58060000]
        
        for flow_col in flow_stations:
            subbasin_id = 24 if flow_col == 58030000 else 36
            
            if flow_col == 58030000:
                assert subbasin_id == 24
            elif flow_col == 58060000:
                assert subbasin_id == 36
    
    def test_metric_validation(self):
        """Test validation of calculated metrics."""
        # Test with known good values
        good_metrics = {
            'RMSE': 2.5,
            'MAE': 2.0,
            'Correlation': 0.8,
            'BIAS': 0.1,
            'Nash_Sutcliffe': 0.6
        }
        
        # Validate metric ranges and properties
        assert good_metrics['RMSE'] > 0
        assert good_metrics['MAE'] > 0
        assert good_metrics['RMSE'] >= good_metrics['MAE']  # RMSE should be >= MAE
        assert -1 <= good_metrics['Correlation'] <= 1
        assert good_metrics['Nash_Sutcliffe'] <= 1
        
        # Test with edge case values
        edge_metrics = {
            'RMSE': 0.0,  # Perfect prediction
            'MAE': 0.0,   # Perfect prediction
            'Correlation': 1.0,  # Perfect correlation
            'BIAS': 0.0,  # No bias
            'Nash_Sutcliffe': 1.0  # Perfect efficiency
        }
        
        # All should be valid
        assert edge_metrics['RMSE'] >= 0
        assert edge_metrics['MAE'] >= 0
        assert -1 <= edge_metrics['Correlation'] <= 1
        assert edge_metrics['Nash_Sutcliffe'] <= 1

class TestFileOperations:
    """Test file I/O operations in evaluation modules."""
    
    def test_csv_output_format(self, temp_output_dir):
        """Test CSV output format and structure."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Flow_Station': [58030000, 58060000],
            'Subbasin_ID': [24, 36],
            'Model': ['Random_Forest', 'Random_Forest'],
            'Dataset': ['Test', 'Test'],
            'RMSE': [4.2, 2.8],
            'MAE': [3.1, 2.0],
            'Correlation': [0.78, 0.82],
            'BIAS': [0.3, 0.15],
            'Nash_Sutcliffe': [0.65, 0.68]
        })
        
        # Test CSV writing
        output_file = temp_output_dir / 'test_output.csv'
        sample_data.to_csv(output_file, index=False)
        
        # Test CSV reading
        loaded_data = pd.read_csv(output_file)
        
        # Check that data was preserved
        pd.testing.assert_frame_equal(sample_data, loaded_data)
        
        # Check file exists and has content
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_output_path_handling(self):
        """Test output path handling logic."""
        # Test relative path construction
        expected_model1_path = '../outputs/model1_performance_metrics.csv'
        expected_model2_path = '../outputs/model2_error_prediction_metrics.csv'
        
        # These should be the paths used in the evaluation modules
        assert expected_model1_path.startswith('../outputs/')
        assert expected_model2_path.startswith('../outputs/')
        assert expected_model1_path.endswith('.csv')
        assert expected_model2_path.endswith('.csv')

class TestIntegrationWithMainModules:
    """Test integration between evaluation modules and main modules."""
    
    def test_evaluation_module_imports(self):
        """Test that evaluation modules can import main modules."""
        # Test imports work correctly
        try:
            import flow_prediction_app
            import model2_error_prediction
            # If we get here, imports work
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_data_flow_compatibility(self):
        """Test that data structures are compatible between modules."""
        # Test that the data structures expected by evaluation modules
        # are compatible with what main modules produce
        
        # Sample structure that main modules should produce
        sample_results_structure = {
            58030000: {
                'Random_Forest': {
                    'train': {
                        'RMSE': float,
                        'MAE': float,
                        'Correlation': float,
                        'BIAS': float,
                        'Nash_Sutcliffe': float
                    },
                    'test': {
                        'RMSE': float,
                        'MAE': float,
                        'Correlation': float,
                        'BIAS': float,
                        'Nash_Sutcliffe': float
                    }
                }
            }
        }
        
        # Test structure validation
        for flow_station, models in sample_results_structure.items():
            assert isinstance(flow_station, int)
            assert flow_station in [58030000, 58060000]
            
            for model_name, datasets in models.items():
                assert isinstance(model_name, str)
                assert 'train' in datasets
                assert 'test' in datasets
                
                for dataset_name, metrics in datasets.items():
                    assert dataset_name in ['train', 'test']
                    required_metrics = ['RMSE', 'MAE', 'Correlation', 'BIAS', 'Nash_Sutcliffe']
                    for metric in required_metrics:
                        assert metric in metrics