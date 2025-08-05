"""Utility functions for model training"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

def encode_categorical_features(X):
    """Encode categorical features"""
    X_encoded = X.copy()
    encoders = {}
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    return X_encoded, encoders

def get_feature_importance(model, feature_names):
    """Get feature importance as a dictionary"""
    importance = model.feature_importances_
    importance_dict = {
        str(feat): float(imp) 
        for feat, imp in sorted(zip(feature_names, importance), 
                               key=lambda x: x[1], reverse=True)[:20]
    }
    return importance_dict