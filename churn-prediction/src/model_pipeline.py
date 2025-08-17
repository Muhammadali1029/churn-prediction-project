"""
Production-ready model pipeline.
Handles training, prediction, and model management.
"""
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

from .config import MODEL_PATH, RANDOM_SEED
from .feature_engineering import ChurnFeatureEngineer
from .model_utils import encode_categorical_features
from .logger_config import setup_logger

logger = setup_logger(__name__)


class ChurnModelPipeline:
    """Production pipeline for churn prediction"""
    
    def __init__(self, model_version: Optional[str] = None):
        self.model_version = model_version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_engineer = ChurnFeatureEngineer()
        self.model_metadata = {}
        self.feature_names = None
        
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> Dict:
        """Train the model and save artifacts"""
        logger.info(f"Starting model training - Version: {self.model_version}")
        
        # Feature engineering
        df_train_features = self.feature_engineer.fit_transform(df_train)
        
        # Prepare features and target
        X_train = df_train_features.drop(['customerID', 'Churn'], axis=1)
        y_train = (df_train_features['Churn'] == 'Yes').astype(int)
        
        # Encode categorical features
        X_train_encoded, self.encoder = encode_categorical_features(X_train)
        self.feature_names = X_train_encoded.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_encoded)
        
        # Train model
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=1000
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training metrics
        train_predictions = self.model.predict_proba(X_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, train_predictions)
        
        metrics = {
            'train_auc': train_auc,
            'train_samples': len(y_train),
            'train_churn_rate': y_train.mean()
        }
        
        # Validation metrics if provided
        if df_val is not None:
            val_metrics = self._evaluate_validation(df_val)
            metrics.update(val_metrics)
        
        # Save model metadata
        self.model_metadata = {
            'version': self.model_version,
            'created_at': datetime.now().isoformat(),
            'model_type': 'LogisticRegression',
            'features': self.feature_names,
            'metrics': metrics,
            'class_weights': 'balanced',
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist()
        }
        
        logger.info(f"Model training completed. AUC: {train_auc:.4f}")
        return metrics
    
    def _evaluate_validation(self, df_val: pd.DataFrame) -> Dict:
        """Evaluate model on validation set"""
        predictions = self.predict_proba(df_val)
        y_val = (df_val['Churn'] == 'Yes').astype(int)
        
        val_auc = roc_auc_score(y_val, predictions)
        
        return {
            'val_auc': val_auc,
            'val_samples': len(y_val),
            'val_churn_rate': y_val.mean()
        }
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Make probability predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Apply same transformations as training
        df_features = self.feature_engineer.transform(df)
        X = df_features.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        
        # Encode categorical features
        X_encoded = X.copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in self.encoder:
                X_encoded[col] = self.encoder[col].transform(X[col].astype(str))
        
        # Ensure same feature order
        X_encoded = X_encoded[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X_encoded)
        
        # Predict
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        probabilities = self.predict_proba(df)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save model artifacts"""
        if path is None:
            path = MODEL_PATH / f"model_{self.model_version}"
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, path / "model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.encoder, path / "encoder.pkl")
        joblib.dump(self.feature_engineer, path / "feature_engineer.pkl")
        
        # Save metadata
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load_model(self, path: Path) -> None:
        """Load model artifacts"""
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        
        self.model = joblib.load(path / "model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")
        self.encoder = joblib.load(path / "encoder.pkl")
        self.feature_engineer = joblib.load(path / "feature_engineer.pkl")
        
        with open(path / "metadata.json", 'r') as f:
            self.model_metadata = json.load(f)
        
        self.model_version = self.model_metadata['version']
        self.feature_names = self.model_metadata['features']
        
        logger.info(f"Model loaded: Version {self.model_version}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'version': self.model_version,
            'metadata': self.model_metadata,
            'is_trained': self.model is not None
        }