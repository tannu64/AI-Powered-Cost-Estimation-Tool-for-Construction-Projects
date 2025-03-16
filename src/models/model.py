import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os


class CostEstimationModel:
    """
    Cost estimation model for construction projects
    """
    
    def __init__(self, model_type='random_forest', **model_params):
        """
        Initialize the cost estimation model
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'xgboost')
            **model_params: Parameters to pass to the model
        """
        self.model_type = model_type.lower()
        self.model_params = model_params
        self.model = None
        self.preprocessor = None
        
        # Initialize the model based on type
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            # Update default parameters with any provided parameters
            default_params.update(model_params)
            self.model = RandomForestRegressor(**default_params)
        
        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            # Update default parameters with any provided parameters
            default_params.update(model_params)
            self.model = xgb.XGBRegressor(**default_params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'random_forest' or 'xgboost'")
    
    def train(self, X_train, y_train, preprocessor=None):
        """
        Train the model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training target values
            preprocessor: Scikit-learn preprocessor for feature transformation
            
        Returns:
            self: Trained model instance
        """
        self.preprocessor = preprocessor
        
        if self.preprocessor:
            X_train_processed = self.preprocessor.fit_transform(X_train)
        else:
            X_train_processed = X_train
        
        self.model.fit(X_train_processed, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            np.array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the model
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not implemented for model type: {self.model_type}")
        
        # If preprocessor is used, we need to get the transformed feature names
        if self.preprocessor and hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        
        # Create DataFrame with feature importances
        if feature_names is not None and len(feature_names) == len(importances):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
        else:
            importance_df = pd.DataFrame({
                'feature_index': range(len(importances)),
                'importance': importances
            })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path, preprocessor_path=None):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
            preprocessor_path (str): Path to save the preprocessor
            
        Returns:
            bool: True if successful
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save the preprocessor if provided
        if self.preprocessor and preprocessor_path:
            joblib.dump(self.preprocessor, preprocessor_path)
        
        return True
    
    @classmethod
    def load_model(cls, model_path, preprocessor_path=None, model_type='random_forest'):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            preprocessor_path (str): Path to the saved preprocessor
            model_type (str): Type of the model being loaded
            
        Returns:
            CostEstimationModel: Loaded model instance
        """
        # Create a new instance
        instance = cls(model_type=model_type)
        
        # Load the model
        instance.model = joblib.load(model_path)
        
        # Load the preprocessor if provided
        if preprocessor_path:
            instance.preprocessor = joblib.load(preprocessor_path)
        
        return instance 