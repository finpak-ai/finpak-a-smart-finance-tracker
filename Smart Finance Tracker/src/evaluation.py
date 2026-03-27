# src/evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", labels=None):
    """Matches Cell 53 & 73"""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels is not None else "auto",
                yticklabels=labels if labels is not None else "auto")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_regression_scatter(y_true, y_pred, title="Regression Actual vs Pred"):
    """Matches Cell 62"""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Matches Cell 63"""
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

def plot_model_errors(errors, targets, title="Model Errors"):
    """Matches Cell 87"""
    plt.figure(figsize=(10, 5))
    plt.bar(targets, errors)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()