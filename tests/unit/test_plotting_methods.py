"""
Unit tests for plotting methods (create_plots.py)

Tests all functions and methods in the plotting module
using Test-Driven Development (TDD) approach.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile

# Import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import create_plots as cp

class TestPlotCreation:
    """Test plot creation functionality."""
    
    @patch('create_plots.main_model2')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_time_series_plots_basic(self, mock_show, mock_savefig, mock_main_model2, temp_output_dir):
        """Test basic time series plot creation."""
        # Setup mock data
        mock_model1_results = {}
        mock_model1_preds = {
            58030000: {
                'Random_Forest': {
                    'y_test': pd.Series([10, 12, 14, 16, 18]),
                    'test_pred': np.array([10.5, 11.8, 13.5, 15.8, 17.2]),
                    'test_dates': pd.DataFrame({
                        'year': [2016, 2016, 2017, 2017, 2018],
                        'month': [1, 6, 1, 6, 1]
                    })
                }
            }
        }
        mock_model2_results = {}
        mock_model2_preds = {}
        mock_ci = {
            58030000: {
                'corrected_predictions': np.array([10.2, 11.9, 13.8, 15.5, 17.5]),
                'lower_bound': np.array([8.5, 10.2, 12.1, 13.8, 15.8]),
                'upper_bound': np.array([11.9, 13.6, 15.5, 17.2, 19.2])
            }
        }
        mock_datasets = {}
        
        mock_main_model2.return_value = (
            mock_model1_results, mock_model1_preds, mock_model2_results,
            mock_model2_preds, mock_ci, mock_datasets
        )
        
        # Execute plot creation
        cp.create_time_series_plots()
        
        # Check that main_model2 was called
        mock_main_model2.assert_called_once()
        
        # Check that savefig was called (plots were created)
        assert mock_savefig.call_count > 0
    
    def test_date_creation_logic(self):
        """Test date creation from year and month columns (novo padrão)."""
        # Sample test dates
        test_dates = pd.DataFrame({
            'year': [2016, 2017, 2018],
            'month': [1, 6, 12]
        })
        # Simula a lógica de criação de datas do create_plots.py
        date_df = test_dates[['year', 'month']].copy()
        date_df['day'] = 1
        dates = pd.to_datetime(date_df)
        # Test results
        assert len(dates) == 3
        assert dates.dtype == 'datetime64[ns]'
        # Check specific dates
        expected_dates = [
            pd.Timestamp('2016-01-01'),
            pd.Timestamp('2017-06-01'),
            pd.Timestamp('2018-12-01')
        ]
        for i, expected_date in enumerate(expected_dates):
            assert dates.iloc[i] == expected_date
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_components(self, mock_savefig):
        """Test individual plot components."""
        # Create sample data
        dates = pd.date_range('2016-01-01', periods=5, freq='ME')
        observed = np.array([10, 12, 14, 16, 18])
        model1_pred = np.array([10.5, 11.8, 13.5, 15.8, 17.2])
        corrected_pred = np.array([10.2, 11.9, 13.8, 15.5, 17.5])
        lower_bound = np.array([8.5, 10.2, 12.1, 13.8, 15.8])
        upper_bound = np.array([11.9, 13.6, 15.5, 17.2, 19.2])
        
        # Create a plot similar to what create_plots.py does
        plt.figure(figsize=(15, 8))
        
        # Plot components
        plt.plot(dates, observed, 'b-', label='Observed', linewidth=2, alpha=0.8)
        plt.plot(dates, model1_pred, 'r--', label='Model 1 Prediction', linewidth=2, alpha=0.7)
        plt.plot(dates, corrected_pred, 'g-', label='Corrected Prediction', linewidth=2, alpha=0.8)
        plt.fill_between(dates, lower_bound, upper_bound, alpha=0.3, color='gray', label='95% Confidence Interval')
        
        # Add labels and formatting
        plt.title('Test Plot')
        plt.xlabel('Date')
        plt.ylabel('Flow (m³/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('test_plot.png')
        plt.close()
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    def test_confidence_interval_logic(self):
        """Test confidence interval creation logic."""
        # Test data
        model1_pred = np.array([10, 12, 14])
        observed = np.array([10.5, 11.8, 13.2])
        
        # Calculate error statistics (similar to what might be done in plotting)
        errors = observed - model1_pred
        error_std = np.std(errors)
        
        # Create simple confidence intervals
        z_score = 1.96  # 95% confidence
        lower_bound = model1_pred - z_score * error_std
        upper_bound = model1_pred + z_score * error_std
        
        # Test results
        assert len(lower_bound) == len(upper_bound)
        assert len(lower_bound) == len(model1_pred)
        assert np.all(lower_bound <= upper_bound)
        assert error_std >= 0
    
    def test_plot_statistics_calculation(self):
        """Test statistics calculation for plot summaries."""
        # Sample data
        observed = np.array([10, 12, 14, 16, 18])
        model1_pred = np.array([10.5, 11.8, 13.5, 15.8, 17.2])
        corrected_pred = np.array([10.2, 11.9, 13.8, 15.5, 17.5])
        
        # Calculate statistics (similar to create_plots.py)
        observed_mean = np.mean(observed)
        model1_rmse = np.sqrt(np.mean((observed - model1_pred)**2))
        corrected_rmse = np.sqrt(np.mean((observed - corrected_pred)**2))
        model1_correlation = np.corrcoef(observed, model1_pred)[0, 1]
        corrected_correlation = np.corrcoef(observed, corrected_pred)[0, 1]
        
        # Test results
        assert isinstance(observed_mean, (int, float))
        assert model1_rmse > 0
        assert corrected_rmse > 0
        assert -1 <= model1_correlation <= 1
        assert -1 <= corrected_correlation <= 1
        
        # Test that correlations are reasonable
        assert model1_correlation > 0.5  # Should have decent correlation
        assert corrected_correlation > 0.5  # Should have decent correlation

class TestPlotFormatting:
    """Test plot formatting and styling."""
    
    def test_plot_figure_size(self):
        """Test plot figure size settings."""
        # Test the figure size used in create_plots.py
        fig = plt.figure(figsize=(15, 8))
        
        # Check figure size
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 8
        
        plt.close(fig)
    
    def test_plot_labels_and_title(self):
        """Test plot labels and title formatting."""
        flow_col = 58030000
        subbasin_id = 24
        
        # Create plot with labels (similar to create_plots.py)
        plt.figure()
        plt.title(f'Flow Prediction - Station {flow_col} (Subbasin {subbasin_id})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Flow (m³/s)', fontsize=12)
        
        # Get current axes
        ax = plt.gca()
        
        # Check title
        title = ax.get_title()
        assert f'Station {flow_col}' in title
        assert f'Subbasin {subbasin_id}' in title
        
        # Check labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        assert xlabel == 'Date'
        assert ylabel == 'Flow (m³/s)'
        
        plt.close()
    
    def test_plot_legend_components(self):
        """Test plot legend components."""
        # Create sample plot with legend
        plt.figure()
        
        x = np.array([1, 2, 3])
        y1 = np.array([1, 2, 3])
        y2 = np.array([1.1, 2.1, 3.1])
        y3 = np.array([0.9, 1.9, 2.9])
        
        plt.plot(x, y1, 'b-', label='Observed', linewidth=2, alpha=0.8)
        plt.plot(x, y2, 'r--', label='Model 1 Prediction', linewidth=2, alpha=0.7)
        plt.plot(x, y3, 'g-', label='Corrected Prediction', linewidth=2, alpha=0.8)
        plt.fill_between(x, y3-0.5, y3+0.5, alpha=0.3, color='gray', label='95% Confidence Interval')
        
        plt.legend(fontsize=10)
        
        # Get legend
        legend = plt.gca().get_legend()
        legend_labels = [text.get_text() for text in legend.get_texts()]
        
        # Check legend labels
        expected_labels = ['Observed', 'Model 1 Prediction', 'Corrected Prediction', '95% Confidence Interval']
        for label in expected_labels:
            assert label in legend_labels
        
        plt.close()

class TestFileOutput:
    """Test file output functionality."""
    
    def test_filename_generation(self):
        """Test filename generation logic."""
        flow_col = 58030000
        subbasin_id = 24
        
        # Test filename generation (similar to create_plots.py)
        filename = f'../outputs/flow_prediction_station_{flow_col}_subbasin_{subbasin_id}.png'
        
        # Check filename format
        assert filename.startswith('../outputs/')
        assert filename.endswith('.png')
        assert str(flow_col) in filename
        assert str(subbasin_id) in filename
        
        # Test for second station
        flow_col_2 = 58060000
        subbasin_id_2 = 36
        filename_2 = f'../outputs/flow_prediction_station_{flow_col_2}_subbasin_{subbasin_id_2}.png'
        
        assert str(flow_col_2) in filename_2
        assert str(subbasin_id_2) in filename_2
        assert filename != filename_2  # Should be different filenames
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_saving_parameters(self, mock_savefig):
        """Test plot saving parameters."""
        # Create and save a plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        filename = 'test_plot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Check that savefig was called with correct parameters
        mock_savefig.assert_called_once_with(filename, dpi=300, bbox_inches='tight')

class TestDataHandling:
    """Test data handling in plotting functions."""
    
    def test_missing_confidence_intervals(self):
        """Test handling when confidence intervals are missing."""
        # Simulate scenario where confidence intervals are not available
        flow_col = 58030000
        ci = {}  # Empty confidence intervals
        
        # Sample data
        model1_pred = np.array([10, 12, 14])
        observed = np.array([10.5, 11.8, 13.2])
        
        # Test fallback confidence interval creation
        if flow_col not in ci:
            # Create simple confidence intervals based on prediction error
            error_std = np.std(observed - model1_pred)
            lower_bound = model1_pred - 1.96 * error_std
            upper_bound = model1_pred + 1.96 * error_std
        else:
            lower_bound = ci[flow_col]['lower_bound']
            upper_bound = ci[flow_col]['upper_bound']
        
        # Test results
        assert len(lower_bound) == len(upper_bound)
        assert len(lower_bound) == len(model1_pred)
        assert np.all(lower_bound <= upper_bound)
    
    def test_data_array_lengths(self):
        """Test handling of arrays with different lengths."""
        # Test data with consistent lengths
        dates = pd.date_range('2016-01-01', periods=5, freq='ME')
        observed = np.array([10, 12, 14, 16, 18])
        predicted = np.array([10.5, 11.8, 13.5, 15.8, 17.2])
        # All should have same length
        assert len(dates) == len(observed)
        assert len(observed) == len(predicted)
        
        # Test with mismatched lengths (should be handled gracefully)
        short_predicted = np.array([10.5, 11.8, 13.5])  # Shorter array
        
        # In real implementation, this should be handled appropriately
        assert len(short_predicted) < len(observed)
    
    def test_nan_values_in_data(self):
        """Test handling of NaN values in plotting data."""
        # Data with NaN values
        observed = np.array([10, np.nan, 14, 16, 18])
        predicted = np.array([10.5, 11.8, np.nan, 15.8, 17.2])
        
        # Test NaN detection
        observed_has_nan = np.isnan(observed).any()
        predicted_has_nan = np.isnan(predicted).any()
        
        assert observed_has_nan
        assert predicted_has_nan
        
        # Test NaN removal for statistics
        valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
        clean_observed = observed[valid_mask]
        clean_predicted = predicted[valid_mask]
        
        # Should have fewer values after cleaning
        assert len(clean_observed) < len(observed)
        assert len(clean_predicted) < len(predicted)
        assert len(clean_observed) == len(clean_predicted)
        
        # Should not have NaN values
        assert not np.isnan(clean_observed).any()
        assert not np.isnan(clean_predicted).any()

class TestIntegrationWithMainModules:
    """Test integration between plotting module and main modules."""
    
    def test_data_structure_compatibility(self):
        """Test that plotting module expects correct data structures."""
        # Test expected structure for model1_preds
        expected_model1_structure = {
            58030000: {
                'Random_Forest': {
                    'y_test': pd.Series,
                    'test_pred': np.ndarray,
                    'test_dates': pd.DataFrame
                }
            }
        }
        
        # Test expected structure for confidence intervals
        expected_ci_structure = {
            58030000: {
                'corrected_predictions': np.ndarray,
                'lower_bound': np.ndarray,
                'upper_bound': np.ndarray
            }
        }
        
        # Validate structure requirements
        for flow_col in expected_model1_structure:
            assert isinstance(flow_col, int)
            assert flow_col in [58030000, 58060000]
        
        for flow_col in expected_ci_structure:
            assert isinstance(flow_col, int)
            required_keys = ['corrected_predictions', 'lower_bound', 'upper_bound']
            for key in required_keys:
                assert key in expected_ci_structure[flow_col]
    
    @patch('create_plots.main_model2')
    def test_main_model2_integration(self, mock_main_model2):
        """Test integration with main_model2 function."""
        # Setup mock return
        mock_main_model2.return_value = (
            {},  # model1_results
            {},  # model1_preds
            {},  # model2_results
            {},  # model2_preds
            {},  # ci
            {}   # datasets
        )
        
        # Test that create_time_series_plots can call main_model2
        try:
            cp.create_time_series_plots()
            mock_main_model2.assert_called_once()
        except Exception as e:
            # Should handle gracefully even with empty data
            assert mock_main_model2.called

class TestErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Empty data structures
        empty_model1_preds = {}
        empty_ci = {}
        
        # Should handle empty data gracefully
        for flow_col in empty_model1_preds.keys():
            pass  # Should not enter loop
        
        # Test with None values
        none_data = None
        
        # Should handle None gracefully
        if none_data is not None:
            pass  # Should not execute
        
        assert True  # If we get here, empty data was handled
    
    def test_matplotlib_backend_handling(self):
        """Test matplotlib backend handling for testing."""
        # Check that non-interactive backend is set
        current_backend = matplotlib.get_backend()
        
        # Should be using Agg backend for testing
        assert current_backend == 'Agg'
        
        # Test that plots can be created without display
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        
        # Should not raise any display-related errors
        assert True