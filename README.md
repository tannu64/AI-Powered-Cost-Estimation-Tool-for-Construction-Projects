# AI-Powered Cost Estimation Tool for Construction Projects

This project implements an AI-powered cost estimation tool for construction projects using machine learning. It provides a web-based interface for users to input project details and receive cost estimates.

## Features

- Machine learning models (Random Forest and XGBoost) for accurate cost prediction
- Interactive web interface for inputting project details
- Detailed model evaluation metrics
- Feature importance analysis
- Comprehensive documentation

## Project Structure

```
.
├── data/                      # Data directory
│   └── construction_data.csv  # Sample construction data
├── models/                    # Saved models directory
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # ML model implementations
│   ├── utils/                 # Utility functions
│   └── web/                   # Web application
├── tests/                     # Test modules
├── construction_cost_estimation.ipynb  # Jupyter notebook for EDA and model training
├── main.py                    # Main script to run the application
└── requirements.txt           # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tannu64/AI-Powered-Cost-Estimation-Tool-for-Construction-Projects.git
cd AI-Powered-Cost-Estimation-Tool-for-Construction-Projects
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv env
.\env\Scripts\activate

# On macOS/Linux
python -m venv env
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Sample Data

```bash
python main.py generate --samples 500 --output data/construction_data.csv
```

### Train a Model

```bash
python main.py train --data data/construction_data.csv --model random_forest --output models
```

### Run the Web Application

```bash
python main.py web --host 127.0.0.1 --port 5000
```

Then open your browser and navigate to `http://127.0.0.1:5000`.

## Jupyter Notebook

The project includes a Jupyter notebook (`construction_cost_estimation.ipynb`) that demonstrates:

1. Data exploration and visualization
2. Model training and evaluation
3. Feature importance analysis
4. Model saving for the web application

To run the notebook:
```bash
jupyter notebook construction_cost_estimation.ipynb
```

## Testing

Run the tests using pytest:
```bash
pytest
```

## Technical Details

### Machine Learning Models

- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees during training.
- **XGBoost**: An optimized gradient boosting library designed to be highly efficient, flexible, and portable.

### Model Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Web Application

The web application is built using Flask and provides a user-friendly interface for inputting project details and receiving cost estimates.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was created as a demonstration of AI-powered cost estimation for construction projects.
- The sample data is synthetic and generated for demonstration purposes only. 