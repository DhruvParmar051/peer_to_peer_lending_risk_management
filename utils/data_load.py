import os
import pandas as pd
import numpy as np

from utils.logger import get_logger 


logger = get_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads merged dataset from CSV."""
    logger.info(f"Loading dataset from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
