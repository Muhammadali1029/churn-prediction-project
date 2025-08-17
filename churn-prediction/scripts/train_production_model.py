"""
Script to train and save production model
"""
import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.model_pipeline import ChurnModelPipeline
from src.config import RAW_DATA_PATH, RANDOM_SEED


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, 
                                        stratify=df['Churn'])
    df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=RANDOM_SEED,
                                       stratify=df_train['Churn'])
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Train model
    print("Training model...")
    pipeline = ChurnModelPipeline(model_version="production")
    metrics = pipeline.train(df_train, df_val)
    
    print(f"Training metrics: {metrics}")
    
    # Save model
    model_path = pipeline.save_model(Path("models/model_production"))
    print(f"Model saved to: {model_path}")
    
    # Test loading
    print("Testing model loading...")
    pipeline_loaded = ChurnModelPipeline()
    pipeline_loaded.load_model(model_path)
    
    # Make test predictions
    test_probs = pipeline_loaded.predict_proba(df_test.head())
    print(f"Test predictions: {test_probs}")
    
    print("✓ Production model ready!")


if __name__ == "__main__":
    main()