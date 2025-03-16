import os
import argparse
import pandas as pd
import numpy as np
from src.data.data_loader import generate_sample_data, load_data, preprocess_data
from src.models.model import CostEstimationModel
from src.web.app import app


def generate_data(n_samples, output_file):
    """Generate sample construction data"""
    print(f"Generating {n_samples} samples of construction data...")
    data = generate_sample_data(n_samples=n_samples, output_file=output_file)
    print(f"Data generated and saved to {output_file}")
    return data


def train_model(data_file, model_type, output_dir):
    """Train a cost estimation model"""
    print(f"Training {model_type} model...")
    
    # Load data
    data = load_data(data_file)
    if data is None:
        print(f"Error: Could not load data from {data_file}")
        return False
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        data, target_column='total_cost', test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = CostEstimationModel(model_type=model_type)
    model.train(X_train, y_train, preprocessor=preprocessor)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'cost_estimation_model.joblib')
    preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
    model.save_model(model_path, preprocessor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
    
    return True


def run_web_app(host, port, debug):
    """Run the web application"""
    print(f"Starting web application on {host}:{port}...")
    app.run(host=host, port=port, debug=debug)


def main():
    """Main function to parse arguments and run the application"""
    parser = argparse.ArgumentParser(description='AI-Powered Cost Estimation Tool for Construction Projects')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate data command
    generate_parser = subparsers.add_parser('generate', help='Generate sample construction data')
    generate_parser.add_argument('--samples', type=int, default=500, help='Number of samples to generate')
    generate_parser.add_argument('--output', type=str, default='data/construction_data.csv', help='Output file path')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a cost estimation model')
    train_parser.add_argument('--data', type=str, default='data/construction_data.csv', help='Data file path')
    train_parser.add_argument('--model', type=str, choices=['random_forest', 'xgboost'], default='random_forest', help='Model type')
    train_parser.add_argument('--output', type=str, default='models', help='Output directory for model files')
    
    # Run web app command
    web_parser = subparsers.add_parser('web', help='Run the web application')
    web_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web app on')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_data(args.samples, args.output)
    
    elif args.command == 'train':
        train_model(args.data, args.model, args.output)
    
    elif args.command == 'web':
        run_web_app(args.host, args.port, args.debug)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 