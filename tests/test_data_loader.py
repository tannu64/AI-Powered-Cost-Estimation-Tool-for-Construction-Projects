import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_data, preprocess_data, generate_sample_data


def test_load_data_success(tmp_path):
    """Test loading data from a valid CSV file"""
    # Create a sample CSV file
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'total_cost': [100, 200, 300]
    })
    
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    
    # Load the data
    loaded_data = load_data(file_path)
    
    # Check if data is loaded correctly
    assert loaded_data is not None
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == 3
    assert list(loaded_data.columns) == ['feature1', 'feature2', 'total_cost']


def test_load_data_failure():
    """Test loading data from a non-existent file"""
    # Try to load a non-existent file
    loaded_data = load_data("non_existent_file.csv")
    
    # Check if None is returned
    assert loaded_data is None


def test_preprocess_data():
    """Test preprocessing data"""
    # Create sample data
    sample_data = pd.DataFrame({
        'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
        'total_cost': [100, 200, 150, 250, 180]
    })
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        sample_data, target_column='total_cost', test_size=0.4, random_state=42
    )
    
    # Check if data is split correctly
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2
    
    # Check if preprocessor is created
    assert preprocessor is not None


def test_preprocess_data_invalid_target():
    """Test preprocessing data with invalid target column"""
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c']
    })
    
    # Try to preprocess with invalid target column
    with pytest.raises(ValueError):
        preprocess_data(sample_data, target_column='non_existent_column')


def test_generate_sample_data():
    """Test generating sample data"""
    # Generate sample data
    n_samples = 50
    sample_data = generate_sample_data(n_samples=n_samples, output_file=None)
    
    # Check if data is generated correctly
    assert sample_data is not None
    assert isinstance(sample_data, pd.DataFrame)
    assert len(sample_data) == n_samples
    
    # Check if all required columns are present
    required_columns = [
        'building_type', 'area_sqm', 'floors', 'location', 'quality_grade',
        'foundation_type', 'roof_type', 'has_basement', 'has_elevator',
        'has_parking', 'labor_rate', 'material_cost_index', 'total_cost'
    ]
    
    for col in required_columns:
        assert col in sample_data.columns


def test_generate_sample_data_with_file(tmp_path):
    """Test generating sample data and saving to file"""
    # Generate sample data and save to file
    output_file = tmp_path / "sample_data.csv"
    sample_data = generate_sample_data(n_samples=10, output_file=output_file)
    
    # Check if file is created
    assert os.path.exists(output_file)
    
    # Load the file and check if data matches
    loaded_data = pd.read_csv(output_file)
    assert len(loaded_data) == 10
    assert list(loaded_data.columns) == list(sample_data.columns)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 