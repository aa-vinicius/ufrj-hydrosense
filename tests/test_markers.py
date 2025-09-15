"""
Test marker examples for HydroSense tests.

This file demonstrates how to use pytest markers to categorize tests.
"""


import pytest
import numpy as np

# Example of marking tests with different categories

@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.model1
def test_example_model1_unit():
    """Example unit test for Model 1."""
    # Simple calculation test
    result = np.mean([1, 2, 3, 4, 5])
    assert result == 3.0

@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.model2
def test_example_model2_unit():
    """Example unit test for Model 2."""
    # Simple error calculation test
    observed = np.array([1, 2, 3])
    predicted = np.array([1.1, 2.1, 2.9])
    errors = observed - predicted
    assert len(errors) == 3

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.pipeline
def test_example_pipeline_integration():
    """Example integration test for pipeline."""
    # Simulate pipeline test
    data = {'values': [1, 2, 3, 4, 5]}
    processed_data = {k: np.mean(v) for k, v in data.items()}
    assert 'values' in processed_data
    assert processed_data['values'] == 3.0

@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.plotting
def test_example_plotting_unit():
    """Example unit test for plotting."""
    # Test plot data preparation
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    assert len(x) == len(y)
    assert np.all(y == 2 * x)

@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.evaluation
def test_example_evaluation_unit():
    """Example unit test for evaluation."""
    # Test metric calculation
    rmse = np.sqrt(np.mean([1, 4, 9]))  # sqrt(mean([1^2, 2^2, 3^2]))
    expected_rmse = np.sqrt(14/3)
    assert abs(rmse - expected_rmse) < 1e-10

@pytest.mark.slow
@pytest.mark.performance
def test_example_performance():
    """Example performance test."""
    import time
    
    # Simulate some computation
    start_time = time.time()
    
    # Simple computation that should be fast
    result = sum(range(1000))
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Should complete quickly
    assert execution_time < 1.0  # Less than 1 second
    assert result == 499500  # Expected sum
