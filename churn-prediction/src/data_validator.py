"""Data validation utilities"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_churn_data(df: pd.DataFrame) -> bool:
    """
    Validate the churn dataset meets our requirements.
    Returns True if valid, raises exception if not.
    """
    issues = []
    
    # Check required columns
    required_cols = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check data types
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        issues.append("TotalCharges is not numeric")
    
    # Check for duplicates
    if df.duplicated(subset=['customerID']).any():
        issues.append(f"Found {df.duplicated(subset=['customerID']).sum()} duplicate customerIDs")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        raise ValueError(f"Data validation failed with {len(issues)} issues")
    
    logger.info("✓ Data validation passed")
    return True