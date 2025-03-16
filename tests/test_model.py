import pytest
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import CostEstimationModel


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y = 10 * X['feature1'] + 5 * X['feature2'] + 2 * X['feature3'] + np.random.normal(0, 0.1, 100)
    return X, y


def test_random_forest_model_init():
    """Test initializing a RandomForest model"""
    model = CostEstimationModel(model_type='random_forest')
    
    assert model.model_type == 'random_forest'
    assert model.model is not None
    assert model.preprocessor is None


def test_xgboost_model_init():
    """Test initializing an XGBoost model"""
    model = CostEstimationModel(model_type='xgboost')
    
    assert model.model_type == 'xgboost'
    assert model.model is not None
    assert model.preprocessor is None


def test_invalid_model_type():
    """Test initializing with an invalid model type"""
    with pytest.raises(ValueError):
        CostEstimationModel(model_type='invalid_model')


def test_model_train(sample_data):
    """Test training a model"""
    X, y = sample_data
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y)
    
    # Check if model is trained
    assert model.model is not None
    
    # Test with preprocessor
    preprocessor = Pipeline([('scaler', StandardScaler())])
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y, preprocessor=preprocessor)
    
    # Check if preprocessor is set
    assert model.preprocessor is not None


def test_model_predict(sample_data):
    """Test making predictions with a trained model"""
    X, y = sample_data
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Check predictions
    assert len(predictions) == len(X)
    assert isinstance(predictions, np.ndarray)


def test_model_predict_without_training(sample_data):
    """Test making predictions with an untrained model"""
    X, y = sample_data
    
    # Create model without training
    model = CostEstimationModel(model_type='random_forest')
    
    # Try to make predictions
    with pytest.raises(ValueError):
        model.predict(X)


def test_model_evaluate(sample_data):
    """Test evaluating a trained model"""
    X, y = sample_data
    
    # Split data
    X_train = X.iloc[:80]
    y_train = y.iloc[:80]
    X_test = X.iloc[80:]
    y_test = y.iloc[80:]
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Check metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics


def test_model_feature_importance(sample_data):
    """Test getting feature importance from a trained model"""
    X, y = sample_data
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y)
    
    # Get feature importance
    importance_df = model.get_feature_importance(feature_names=X.columns)
    
    # Check importance DataFrame
    assert isinstance(importance_df, pd.DataFrame)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert len(importance_df) == len(X.columns)


def test_model_save_load(sample_data, tmp_path):
    """Test saving and loading a model"""
    X, y = sample_data
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    model.save_model(model_path)
    
    # Check if file is created
    assert os.path.exists(model_path)
    
    # Load model
    loaded_model = CostEstimationModel.load_model(model_path, model_type='random_forest')
    
    # Check loaded model
    assert loaded_model.model is not None
    assert loaded_model.model_type == 'random_forest'
    
    # Make predictions with loaded model
    predictions = loaded_model.predict(X)
    assert len(predictions) == len(X)


def test_model_save_load_with_preprocessor(sample_data, tmp_path):
    """Test saving and loading a model with preprocessor"""
    X, y = sample_data
    
    # Create preprocessor
    preprocessor = Pipeline([('scaler', StandardScaler())])
    
    # Create and train model
    model = CostEstimationModel(model_type='random_forest', n_estimators=10)
    model.train(X, y, preprocessor=preprocessor)
    
    # Save model and preprocessor
    model_path = tmp_path / "model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    model.save_model(model_path, preprocessor_path)
    
    # Check if files are created
    assert os.path.exists(model_path)
    assert os.path.exists(preprocessor_path)
    
    # Load model and preprocessor
    loaded_model = CostEstimationModel.load_model(
        model_path, preprocessor_path=preprocessor_path, model_type='random_forest'
    )
    
    # Check loaded model and preprocessor
    assert loaded_model.model is not None
    assert loaded_model.preprocessor is not None
    
    # Make predictions with loaded model
    predictions = loaded_model.predict(X)
    assert len(predictions) == len(X)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 