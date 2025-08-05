"""
Feature engineering for churn prediction.
Following insights from EDA.
"""
import pandas as pd
from logger_config import setup_logger

logger = setup_logger(__name__)

class ChurnFeatureEngineer:
    """
    Industry practice: Use classes for stateful transformations
    This ensures train/test consistency
    """
    
    def __init__(self):
        self.feature_names = []
        self.categorical_mappings = {}
    
    def fit(self, df: pd.DataFrame) -> 'ChurnFeatureEngineer':
        """Learn any parameters needed for transformation"""
        # Store categorical mappings for consistent encoding
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'customerID':
                self.categorical_mappings[col] = df[col].unique()
        return self
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric columns that might have been read as strings.
        This is a common issue in real-world data.
        """
        # Convert TotalCharges to numeric, replacing errors with NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Handle missing values appropriately
        # For TotalCharges, if missing, we can impute based on MonthlyCharges * tenure
        mask = df['TotalCharges'].isna()
        if mask.any():
            logger.warning(f"Found {mask.sum()} rows with invalid TotalCharges. Imputing...")
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
            # If tenure is 0, just use 0
            df.loc[mask & (df['tenure'] == 0), 'TotalCharges'] = 0
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        df = df.copy()

        # Clean data first
        df = self._clean_numeric_columns(df)
        
        # 1. Tenure-based features (from eda insights)
        df['tenure_bucket'] = pd.cut(df['tenure'], 
                                     bins=[-1, 6, 12, 24, 48, 72], 
                                     labels=['0-6m', '6-12m', '1-2y', '2-4y', '4y+'])
        df['is_new_customer'] = (df['tenure'] <= 12).astype(float)
        df['tenure_squared'] = df['tenure'] ** 2  # Capture non-linear effects
        
        # 2. Revenue features
        # Convert TotalHCarges to numeric to avoid getting an error
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        df['avg_charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # 3. Service portfolio features (from eda insight about unprotected customers)
        protection_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['num_protection_services'] = sum(
            (df[service] == 'Yes').astype(int) for service in protection_services
        )
        df['has_premium_unprotected'] = (
            (df['InternetService'] == 'Fiber optic') & 
            (df['num_protection_services'] == 0)
        ).astype(float)
        
        # 4. Contract value features
        df['contract_value_score'] = df['Contract'].map({
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        })
        
        # 5. Payment risk features
        df['is_electronic_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(float)
        
        # 6. Additional features based on business logic
        df['revenue_per_service'] = df['MonthlyCharges'] / (df['num_protection_services'] + 1)
        df['high_value_customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(float)
        
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['customerID']])} features")
        return df
    
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)