import os
import numpy as np
import pandas as pd
import warnings

from utils.data_load import load_data
from utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target column 'is_default' from 'loan_status'.
    Keeps only rows where loan_status is Fully Paid, Charged Off, or Default.
    """
    logger.info("Creating target variable 'is_default'.")

    valid_status = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(valid_status)].copy()

    df["is_default"] = df["loan_status"].apply(
        lambda x: 1 if x in ["Charged Off", "Default"] else 0
    )

    df.drop(columns=["loan_status"], inplace=True)

    logger.info(f"Target column created. Data shape after filtering: {df.shape}")
    return df


def feature_engineering_pipeline(input_path: str, output_dir: str):
    """
    Main Feature Engineering Pipeline (No PCA).
    
    Steps:
    1. Load cleaned data
    2. Create binary target variable
    3. Save the dataset (no PCA, no dimensionality reduction)
    """
    logger.info("Starting feature engineering pipeline.")

    # Load cleaned dataset
    df = load_data(input_path)
    logger.info(f"Data loaded successfully: {df.shape}")

    # Create binary target
    df = create_target(df)

    # Save final dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "engineered_data.parquet")
    df.to_parquet(output_path, index=False)

    logger.info(f"Feature engineered data saved to: {output_path}")
    logger.info("Feature engineering pipeline completed successfully.")
