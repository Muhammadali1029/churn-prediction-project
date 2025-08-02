"""
Data download utilities.
Industry practice: Version control your data acquisition
"""
import os
import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_telco_churn(output_path: Path) -> Optional[Path]:
    """
    Download Telco churn dataset.
    In production, this would connect to the data warehouse.
    """
    # For now, manually download from Kaggle and place in data/raw/
    # URL: https://www.kaggle.com/blastchar/telco-customer-churn
    
    expected_file = output_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if expected_file.exists():
        logger.info(f"Data already exists at {expected_file}")
        # Industry practice: Verify data integrity
        df = pd.read_csv(expected_file)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return expected_file
    else:
        logger.error(
            f"Please download the Telco Customer Churn dataset from Kaggle "
            f"and place it in {output_path}"
        )
        return None

if __name__ == "__main__":
    from config import RAW_DATA_PATH
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    download_telco_churn(RAW_DATA_PATH)
