"""
Unit tests for model pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.model_pipeline import ChurnModelPipeline
from src.config import RANDOM_SEED


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(RANDOM_SEED)
    n_samples = 100
    
    data = pd.DataFrame({
        'customerID': [f'CUST-{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 100, n_samples),
        'TotalCharges': np.random.uniform(20, 7000, n_samples).astype(str),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.26, 0.74])
    })
    
    # Add other required columns with default values
    for col in ['Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']:
        if col not in data.columns:
            data[col] = 'No'
    
    return data


class TestChurnModelPipeline:
    """Test cases for ChurnModelPipeline"""
    
    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = ChurnModelPipeline()
        assert pipeline.model is None
        assert pipeline.model_version is not None
        
    def test_training(self, sample_data):
        """Test model training"""
        pipeline = ChurnModelPipeline()
        
        # Split data
        train_data = sample_data.iloc[:80]
        val_data = sample_data.iloc[80:]
        
        # Train model
        metrics = pipeline.train(train_data, val_data)
        
        # Check metrics
        assert 'train_auc' in metrics
        assert 'val_auc' in metrics
        assert 0 <= metrics['train_auc'] <= 1
        assert 0 <= metrics['val_auc'] <= 1
        
        # Check model is trained
        assert pipeline.model is not None
        assert pipeline.scaler is not None
        assert pipeline.encoder is not None
        
    def test_prediction(self, sample_data):
        """Test prediction functionality"""
        pipeline = ChurnModelPipeline()
        
        # Train model
        pipeline.train(sample_data)
        
        # Test predictions
        test_sample = sample_data.iloc[:10]
        probabilities = pipeline.predict_proba(test_sample)
        
        # Check predictions
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)
        
        # Test binary predictions
        predictions = pipeline.predict(test_sample, threshold=0.5)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
        
    def test_save_load_model(self, sample_data):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_model"
            
            # Train and save
            pipeline1 = ChurnModelPipeline()
            pipeline1.train(sample_data)
            original_probs = pipeline1.predict_proba(sample_data.iloc[:5])
            pipeline1.save_model(temp_path)
            
            # Load and predict
            pipeline2 = ChurnModelPipeline()
            pipeline2.load_model(temp_path)
            loaded_probs = pipeline2.predict_proba(sample_data.iloc[:5])
            
            # Check predictions are same
            np.testing.assert_array_almost_equal(original_probs, loaded_probs)
            
    def test_metadata_tracking(self, sample_data):
        """Test metadata is properly tracked"""
        pipeline = ChurnModelPipeline()
        pipeline.train(sample_data)
        
        metadata = pipeline.model_metadata
        assert 'version' in metadata
        assert 'created_at' in metadata
        assert 'model_type' in metadata
        assert 'features' in metadata
        assert 'metrics' in metadata
        
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        pipeline = ChurnModelPipeline()
        
        # Test prediction without training
        with pytest.raises(ValueError):
            pipeline.predict_proba(pd.DataFrame())
            
    def test_feature_consistency(self, sample_data):
        """Test feature handling consistency"""
        pipeline = ChurnModelPipeline()
        pipeline.train(sample_data)
        
        # Create data with missing features
        incomplete_data = sample_data[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']].copy()
        
        # Should raise KeyError for missing features
        with pytest.raises(KeyError) as excinfo:
            pipeline.predict_proba(incomplete_data)
        
        # The error will be about a missing column (like 'OnlineSecurity')
        # This verifies the pipeline properly validates required features
        error_msg = str(excinfo.value).lower()
        
        # Check that it's complaining about a missing feature
        # The actual error will be the name of a missing column
        assert any(feature.lower() in error_msg for feature in [
            'onlinesecurity', 'onlinebackup', 'deviceprotection', 
            'techsupport', 'contract', 'paperlessbilling'
        ])