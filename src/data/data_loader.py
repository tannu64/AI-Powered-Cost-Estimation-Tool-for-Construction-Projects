import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """
    Load construction project data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data, target_column='total_cost', test_size=0.2, random_state=42):
    """
    Preprocess the construction data for model training
    
    Args:
        data (pd.DataFrame): Raw construction data
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    if data is None or target_column not in data.columns:
        raise ValueError(f"Invalid data or target column '{target_column}' not found")
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


def generate_sample_data(n_samples=100, output_file='construction_data.csv'):
    """
    Generate sample construction project data for testing
    
    Args:
        n_samples (int): Number of samples to generate
        output_file (str): Path to save the generated data
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        'building_type': np.random.choice(['Residential', 'Commercial', 'Industrial'], n_samples),
        'area_sqm': np.random.uniform(100, 10000, n_samples),
        'floors': np.random.randint(1, 50, n_samples),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'quality_grade': np.random.choice(['Standard', 'Premium', 'Luxury'], n_samples),
        'foundation_type': np.random.choice(['Concrete', 'Pile', 'Slab'], n_samples),
        'roof_type': np.random.choice(['Flat', 'Pitched', 'Dome'], n_samples),
        'has_basement': np.random.choice([0, 1], n_samples),
        'has_elevator': np.random.choice([0, 1], n_samples),
        'has_parking': np.random.choice([0, 1], n_samples),
        'labor_rate': np.random.uniform(20, 50, n_samples),
        'material_cost_index': np.random.uniform(0.8, 1.5, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (total cost) based on features
    # This is a simplified model for demonstration
    base_cost = 500  # base cost per sqm
    
    # Building type factor
    building_type_factor = df['building_type'].map({
        'Residential': 1.0,
        'Commercial': 1.3,
        'Industrial': 1.1
    })
    
    # Location factor
    location_factor = df['location'].map({
        'Urban': 1.2,
        'Suburban': 1.0,
        'Rural': 0.8
    })
    
    # Quality factor
    quality_factor = df['quality_grade'].map({
        'Standard': 1.0,
        'Premium': 1.5,
        'Luxury': 2.0
    })
    
    # Calculate total cost with some randomness
    df['total_cost'] = (
        base_cost * df['area_sqm'] * 
        building_type_factor * 
        location_factor * 
        quality_factor * 
        (1 + 0.1 * df['floors']) *
        df['material_cost_index'] *
        (1 + 0.05 * df['has_basement']) *
        (1 + 0.03 * df['has_elevator']) *
        (1 + 0.02 * df['has_parking'])
    )
    
    # Add some noise to make it more realistic
    df['total_cost'] = df['total_cost'] * np.random.normal(1, 0.1, n_samples)
    
    # Save to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Sample data saved to {output_file}")
    
    return df 