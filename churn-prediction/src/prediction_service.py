"""
Prediction service for real-time inference.
Handles single and batch predictions with business logic.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

from model_pipeline import ChurnModelPipeline
from business_optimiser import ChurnBusinessOptimiser
from logger_config import setup_logger

logger = setup_logger(__name__)


class ChurnPredictionService:
    """Service for making churn predictions with business logic"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 business_threshold: float = 0.3):
        self.pipeline = ChurnModelPipeline()
        
        if model_path:
            self.pipeline.load_model(model_path)
        
        self.business_threshold = business_threshold
        self.optimizer = ChurnBusinessOptimiser()
        
    def predict_single(self, customer_data: Dict) -> Dict:
        """Predict for a single customer"""
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Get prediction
        probability = self.pipeline.predict_proba(df)[0]
        will_churn = probability >= self.business_threshold
        
        # Business recommendation
        if will_churn:
            retention_value = self._calculate_retention_value(customer_data)
            recommendation = "High priority for retention" if retention_value > 100 else "Monitor closely"
        else:
            recommendation = "Low churn risk"
        
        return {
            'customer_id': customer_data.get('customerID'),
            'churn_probability': float(probability),
            'will_churn': bool(will_churn),
            'recommendation': recommendation,
            'predicted_at': datetime.now().isoformat()
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple customers"""
        logger.info(f"Making batch predictions for {len(df)} customers")
        
        # Get predictions
        probabilities = self.pipeline.predict_proba(df)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customerID': df['customerID'],
            'churn_probability': probabilities,
            'will_churn': probabilities >= self.business_threshold,
            'risk_segment': pd.cut(probabilities, 
                                  bins=[0, 0.2, 0.5, 0.8, 1.0],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        })
        
        # Add business metrics
        results['monthly_revenue'] = df['MonthlyCharges']
        results['annual_revenue_at_risk'] = results['monthly_revenue'] * 12 * results['churn_probability']
        results['retention_priority'] = results['annual_revenue_at_risk'].rank(ascending=False)
        
        return results
    
    def _calculate_retention_value(self, customer_data: Dict) -> float:
        """Calculate expected value of retaining a customer"""
        monthly_revenue = customer_data.get('MonthlyCharges', 0)
        return monthly_revenue * 12 * 0.3  # Assuming 30% retention success
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update business threshold based on campaign performance"""
        old_threshold = self.business_threshold
        self.business_threshold = new_threshold
        logger.info(f"Updated threshold from {old_threshold:.3f} to {new_threshold:.3f}")