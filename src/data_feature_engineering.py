import os
import numpy as np
import pandas as pd
import logging
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.data_load import load_data

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target column 'is_default' from 'loan_status'.
    Keeps only rows where loan_status is Fully Paid, Charged Off, or Default.
    """
    logger.info("Creating target variable 'is_default'.")
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
    df.drop(columns=["loan_status"], inplace=True)
    logger.info(f"Target column created. Data shape after filtering: {df.shape}")
    return df


def apply_pca(df: pd.DataFrame, variance_threshold: float = 0.95) -> pd.DataFrame:
    """
    Apply standard scaling and PCA to numeric columns.
    Retains components that explain the given variance threshold (default 95%).
    """
    logger.info("Applying PCA to numeric features.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target variable if present
    if "is_default" in numeric_cols:
        numeric_cols.remove("is_default")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components=variance_threshold, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    pca_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=df.index)

    df_pca = pd.concat([pca_df, df.drop(columns=numeric_cols)], axis=1)

    logger.info(f"PCA applied. Retained {pca_df.shape[1]} components explaining {variance_threshold*100}% variance.")
    return df_pca


def feature_engineering_pipeline(input_path: str, output_dir: str):
    """
    Main feature engineering pipeline.
    1. Load cleaned data
    2. Create binary target variable
    3. Apply PCA on numeric features
    4. Save the final dataset
    """
    logger.info("Starting feature engineering pipeline.")
    df = load_data(input_path)
    logger.info(f"Data loaded successfully: {df.shape}")

    df = create_target(df)
    df_pca = apply_pca(df)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "engineered_data.parquet")
    df_pca.to_parquet(output_path, index=False)

    logger.info(f"Feature engineered data saved to: {output_path}")
    logger.info("Feature engineering pipeline completed successfully.")
