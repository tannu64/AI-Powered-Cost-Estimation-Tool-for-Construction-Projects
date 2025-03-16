import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_feature_importance(importance_df, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importances
        top_n (int): Number of top features to show
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get top N features
    if len(importance_df) > top_n:
        plot_df = importance_df.head(top_n)
    else:
        plot_df = importance_df
    
    # Create horizontal bar plot
    sns.barplot(x='importance', y='feature', data=plot_df)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt.gcf()


def plot_prediction_vs_actual(y_true, y_pred, figsize=(10, 6)):
    """
    Plot predicted vs actual values
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    
    return plt.gcf()


def plot_residuals(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        figsize (tuple): Figure size
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot of residuals
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    
    # Add histogram of residuals
    plt.figure(figsize=figsize)
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residual Value')
    plt.tight_layout()
    
    return plt.gcf()


def plot_correlation_matrix(data, figsize=(12, 10)):
    """
    Plot correlation matrix of features
    
    Args:
        data (pd.DataFrame): DataFrame with features
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr = data.select_dtypes(include=['int64', 'float64']).corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return plt.gcf()


def plot_feature_distributions(data, figsize=(15, 10)):
    """
    Plot distributions of numeric features
    
    Args:
        data (pd.DataFrame): DataFrame with features
        figsize (tuple): Figure size
    """
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    n_cols = len(numeric_cols)
    
    if n_cols == 0:
        return None
    
    # Calculate grid dimensions
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_categorical_counts(data, figsize=(15, 10)):
    """
    Plot count plots for categorical features
    
    Args:
        data (pd.DataFrame): DataFrame with features
        figsize (tuple): Figure size
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    n_cols = len(categorical_cols)
    
    if n_cols == 0:
        return None
    
    # Calculate grid dimensions
    n_rows = (n_cols + 1) // 2  # 2 columns per row
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    # Handle case with only one categorical column
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            sns.countplot(y=col, data=data, ax=axes[i])
            axes[i].set_title(f'Count of {col}')
            axes[i].set_ylabel(col)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    return fig 