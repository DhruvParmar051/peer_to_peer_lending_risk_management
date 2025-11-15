import os
import pandas as pd
import numpy as np
<<<<<<< HEAD
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
=======

from utils.logger import get_logger 


logger = get_logger(__name__)
>>>>>>> 9a1af4c (created new logger module)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads merged dataset from CSV."""
    logger.info(f"Loading dataset from {file_path}")
<<<<<<< HEAD
<<<<<<< HEAD
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
<<<<<<< HEAD
    return df
=======
=======
    
>>>>>>> 9cc7e8d (updates)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

>>>>>>> 81b372d (data cleaning created)
=======
    return df
>>>>>>> 3ea0aba (logs created)
