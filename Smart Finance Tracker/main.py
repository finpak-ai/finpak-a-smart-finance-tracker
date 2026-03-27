# main.py

import pandas as pd
from sklearn.pipeline import Pipeline
from src import config, processing, modelling

def main():
    # 1. Load Data
    data = pd.read_csv(config.DATA_PATH)

    # 2. Preprocessing Pipeline
    # (Reuse the pipeline from the previous answer for src/processing.py)
    preprocess_pipe = Pipeline([
        ('label_encoder', processing.CustomLabelEncoder(cols=config.CAT_COLS)),
        ('outlier', processing.OutlierCapper(cols=config.NUM_COLS)),
        ('feat_eng', processing.FeatureEngineer(config.WEEKLY_WEIGHTS, config.EXPENSE_COLS))
    ])
    
    print("Processing Data...")
    df_processed = preprocess_pipe.fit_transform(data)

    # 3. Train All 4 Categories of Models
    
    # --- Category 1: Overspend ---
    print("\n=== 1. Overspend Prediction ===")
    t1 = modelling.OverspendTrainer(config.FEATURES_OVERSPEND, config.TARGET_OVERSPEND)
    t1.train_and_evaluate(df_processed)
    t1.save_models()

    # --- Category 2: Actual Savings ---
    print("\n=== 2. Savings Prediction ===")
    t2 = modelling.SavingsTrainer(config.FEATURES_SAVINGS, config.TARGET_SAVINGS)
    t2.train_and_evaluate(df_processed)
    t2.save_models()

    # --- Category 3: Financial Health ---
    print("\n=== 3. Financial Health ===")
    # Note: Ensure Target is encoded (Good=0, etc.) in df_processed or inside Trainer
    t3 = modelling.HealthTrainer(config.FEATURES_HEALTH, config.TARGET_HEALTH)
    t3.train_and_evaluate(df_processed)
    t3.save_models()

    # --- Category 4: Suggestions (Multi-Output) ---
    print("\n=== 4. Suggesting Savings ===")
    t4 = modelling.SuggestionsTrainer(config.FEATURES_SUGGESTIONS, config.TARGETS_SUGGESTIONS)
    t4.train_and_evaluate(df_processed)
    t4.save_models()

if __name__ == "__main__":
    main()