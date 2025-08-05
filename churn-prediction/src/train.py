"""
Model training pipeline with experiment tracking.
Industry practice: Every model run should be tracked and reproducible.
"""
import argparse
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import json
from datetime import datetime

from config import RANDOM_SEED, MODEL_PATH, PROCESSED_DATA_PATH
from feature_engineering import ChurnFeatureEngineer
from model_utils import encode_categorical_features, get_feature_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """Handles model training with MLflow tracking"""
    
    def __init__(self, experiment_name="churn_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.feature_engineer = ChurnFeatureEngineer()
        
    def prepare_data(self, df):
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Apply feature engineering
        df_features = self.feature_engineer.fit_transform(df)
        
        # Separate features and target
        X = df_features.drop(['customerID', 'Churn'], axis=1)
        y = (df_features['Churn'] == 'Yes').astype(int)
        
        # Encode categorical features
        X_encoded, encoder = encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
        )
        
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, encoder
    
    def train_baseline(self, X_train, X_val, y_train, y_val):
        """Train baseline logistic regression model"""
        with mlflow.start_run(run_name="baseline_logistic_regression"):
            logger.info("Training baseline model...")
            
            # Log parameters
            mlflow.log_param("model_type", "logistic_regression")
            mlflow.log_param("random_state", RANDOM_SEED)
            
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Metrics
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("val_auc", auc)
            mlflow.log_metric("val_accuracy", (y_pred == y_val).mean())
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Baseline AUC: {auc:.4f}")
            
            return model, scaler, auc
    
    def train_xgboost(self, X_train, X_val, y_train, y_val, params=None):
        """Train XGBoost model with hyperparameter tracking"""
        with mlflow.start_run(run_name="xgboost"):
            logger.info("Training XGBoost model...")
            
            # Default parameters
            if params is None:
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': RANDOM_SEED,
                    'eval_metric': 'auc'
                }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = xgb.XGBClassifier(**params)
            
            # Fit with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("val_auc", auc)
            mlflow.log_metric("val_accuracy", (y_pred == y_val).mean())
            mlflow.log_metric("best_iteration", model.best_iteration)
            
            # Feature importance
            importance_dict = get_feature_importance(model, X_train.columns)
            mlflow.log_dict(importance_dict, "feature_importance.json")
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            logger.info(f"XGBoost AUC: {auc:.4f}")
            
            return model, auc
    
    def hyperparameter_search(self, X_train, X_val, y_train, y_val):
        """Simple hyperparameter search with tracking"""
        logger.info("Starting hyperparameter search...")
        
        param_grid = [
            {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
            {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200},
            {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 300},
        ]
        
        best_auc = 0
        best_params = None
        
        for params in param_grid:
            full_params = {
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_SEED,
                'eval_metric': 'auc',
                **params
            }
            
            _, auc = self.train_xgboost(X_train, X_val, y_train, y_val, full_params)
            
            if auc > best_auc:
                best_auc = auc
                best_params = full_params
        
        logger.info(f"Best AUC: {best_auc:.4f} with params: {best_params}")
        return best_params, best_auc

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model-type', type=str, default='all', choices=['baseline', 'xgboost', 'all'])
    parser.add_argument('--hyperparam-search', action='store_true', help='Run hyperparameter search')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = trainer.prepare_data(df)
    
    # Train models based on selection
    if args.model_type in ['baseline', 'all']:
        trainer.train_baseline(X_train, X_val, y_train, y_val)
    
    if args.model_type in ['xgboost', 'all']:
        if args.hyperparam_search:
            best_params, _ = trainer.hyperparameter_search(X_train, X_val, y_train, y_val)
            # Train final model with best params
            trainer.train_xgboost(X_train, X_val, y_train, y_val, best_params)
        else:
            trainer.train_xgboost(X_train, X_val, y_train, y_val)
    
    logger.info("Training complete! Check MLflow UI for results.")

if __name__ == "__main__":
    main()