# src/modelling.py

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error  # <--- THIS WAS MISSING

# Models Task 1 (Overspend)
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Models Task 2 (Savings)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Models Task 3 (Health)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Models Task 4 (Suggestions)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from src import evaluation

class BaseTrainer:
    def __init__(self, name, features, target):
        self.name = name
        self.features = features
        self.target = target
        self.models = {}
        self.scalers = {}

    def prepare_data(self, df):
        # Handle missing cols gracefully or assume preprocessing is done
        X = df[self.features].fillna(0)
        y = df[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def save_models(self, path="saved_models"):
        if not os.path.exists(path): os.makedirs(path)
        for model_name, model in self.models.items():
            file_name = f"{self.name}_{model_name.replace(' ', '_')}.pkl"
            with open(os.path.join(path, file_name), "wb") as f:
                pickle.dump(model, f)
            print(f"Saved {file_name}")

# --- 1. Overspend Prediction (Classification) ---
class OverspendTrainer(BaseTrainer):
    def __init__(self, features, target):
        super().__init__("Overspend", features, target)
        self.models = {
            "Perceptron": make_pipeline(StandardScaler(), Perceptron(max_iter=1000, random_state=42)),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

    def train_and_evaluate(self, df):
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        for name, model in self.models.items():
            print(f"Training {self.name} - {name}...")
            model.fit(X_train, y_train)
            
            # Evaluation
            preds = model.predict(X_test)
            acc = np.mean(preds == y_test)
            print(f"Accuracy: {acc:.4f}")
            
            # Plot Confusion Matrix
            evaluation.plot_confusion_matrix(y_test, preds, title=f"CM {name} Overspend")

# --- 2. Actual Savings Prediction (Regression) ---
class SavingsTrainer(BaseTrainer):
    def __init__(self, features, target):
        super().__init__("Savings", features, target)
        self.models = {
            "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42)
        }

    def train_and_evaluate(self, df):
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        for name, model in self.models.items():
            print(f"Training {self.name} - {name}...")
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            print(f"R2 Score: {model.score(X_test, y_test):.4f}")
            
            evaluation.plot_regression_scatter(y_test, preds, title=f"Scatter {name} Savings")
            
            # Feature Importance for Tree models
            if "Tree" in name or "Forest" in name:
                if hasattr(model, 'feature_importances_'):
                    evaluation.plot_feature_importance(model, self.features, title=f"Features {name}")

# --- 3. Financial Health (Multiclass) ---
class HealthTrainer(BaseTrainer):
    def __init__(self, features, target):
        super().__init__("Health", features, target)
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB())
        }

    def train_and_evaluate(self, df):
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        for name, model in self.models.items():
            print(f"Training {self.name} - {name}...")
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            acc = np.mean(preds == y_test)
            print(f"Accuracy: {acc:.4f}")
            
            evaluation.plot_confusion_matrix(y_test, preds, title=f"CM {name} Health")

# --- 4. Suggesting Savings (Multi-Output) ---
class SuggestionsTrainer(BaseTrainer):
    def __init__(self, features, target):
        super().__init__("Suggestions", features, target)
        self.models = {
            "Multi Linear": MultiOutputRegressor(LinearRegression()),
            "Multi Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=8)),
            "Multi KNN": MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5))
        }

    def train_and_evaluate(self, df):
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        for name, model in self.models.items():
            print(f"Training {self.name} - {name}...")
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            
            # Calculate errors for plotting
            # Now mean_absolute_error is defined!
            errors = [mean_absolute_error(y_test[col], preds[:, i]) for i, col in enumerate(self.target)]
            evaluation.plot_model_errors(errors, self.target, title=f"Errors {name}")