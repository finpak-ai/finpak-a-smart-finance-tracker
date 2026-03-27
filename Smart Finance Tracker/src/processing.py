# src/processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.quantiles_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                self.quantiles_[col] = {
                    'q1': X[col].quantile(0.01),
                    'q99': X[col].quantile(0.99)
                }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns and col in self.quantiles_:
                q1 = self.quantiles_[col]['q1']
                q99 = self.quantiles_[col]['q99']
                X[col] = X[col].clip(q1, q99)
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Robust Label Encoder: Maps unseen labels to the most frequent (mode) value
    to prevent crashes during production use.
    """
    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}
        self.modes_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            le = LabelEncoder()
            # Fit on strings to handle mixed types
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
            self.modes_[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in self.encoders_:
                le = self.encoders_[col]
                fallback_value = str(self.modes_[col])
                known_classes = set(le.classes_)

                # Helper to swap unknown values with the mode
                def safe_map(val):
                    s_val = str(val)
                    if s_val in known_classes:
                        return s_val
                    return fallback_value

                cleaned_col = X[col].apply(safe_map)
                X[col + 'encoded'] = le.transform(cleaned_col)
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, weights, expense_cols):
        self.weights = weights
        self.expense_cols = expense_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Auto-Calculate Disposable Income if missing
        if 'Disposable_Income' not in X.columns:
            X['Disposable_Income'] = X['Income'] 

        # 2. Total Expenses
        X['Total_Expenses'] = X[self.expense_cols].sum(axis=1)

        # 3. Actual Savings & Overspend
        X['Actual_Savings'] = X['Disposable_Income'] - X['Total_Expenses']
        X['Overspend'] = (X['Actual_Savings'] < 0).astype(int)

        # 4. Weekly Breakdown (W1-W4)
        for col in self.expense_cols:
            for w, v in self.weights.items():
                X[f'{col}_{w}'] = X[col] * v

        # 5. Weekly Totals
        for w in self.weights:
            weekly_sub_cols = [f'{c}_{w}' for c in self.expense_cols]
            X[f'Week_{w}_Total'] = X[weekly_sub_cols].sum(axis=1)

        # 6. Spending Trend (Difference between Week 4 and Week 1 Expenses)
        X['Spending_Trend'] = X['Week_W4_Total'] - X['Week_W1_Total']

        # 7. Savings Efficiency
        if 'Desired_Savings' in X.columns:
            X['Savings_Efficiency'] = X['Actual_Savings'] / (X['Desired_Savings'] + 1.0)

        # 8. Weekly Income & Weekly Savings
        X['Weekly_Income'] = X['Income'] / 4.0
        weekly_savings_cols = []
        for w in ['W1', 'W2', 'W3', 'W4']:
             col_name = f'Weekly_Savings_{w}'
             X[col_name] = X['Weekly_Income'] - X[f'Week_{w}_Total']
             weekly_savings_cols.append(col_name)

        # 9. Weekly Overspend Trend (THE FIX)
        # Calculates the count of weeks where savings were negative.
        # This creates the missing 'Weekly_Overspend_Trend' column.
        X['Weekly_Overspend_Trend'] = (X[weekly_savings_cols] < 0).sum(axis=1)
        
        return X