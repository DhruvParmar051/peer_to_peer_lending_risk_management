import os
import pandas as pd
import numpy as np
<<<<<<< HEAD

from utils.logger import get_logger 


logger = get_logger(__name__)
=======
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
>>>>>>> 81b372d (data cleaning created)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads merged dataset from CSV."""
    logger.info(f"Loading dataset from {file_path}")
<<<<<<< HEAD
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
=======
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=True, compression='gzip')
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

>>>>>>> 81b372d (data cleaning created)
