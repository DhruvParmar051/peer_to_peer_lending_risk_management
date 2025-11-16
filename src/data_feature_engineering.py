# src/data_feature_engineering.py
import os
import numpy as np
import pandas as pd
from utils.data_load import load_data
from utils.logger import get_logger

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = get_logger(__name__)

def create_target_and_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target 'is_default' and a few safe derived features from date columns.
    """
    logger.info("Creating target variable 'is_default'.")
    valid_status = ["Fully Paid", "Charged Off", "Default"]
    if "loan_status" in df.columns:
        df = df[df["loan_status"].isin(valid_status)].copy()
        df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
        df = df.drop(columns=["loan_status"])
    else:
        logger.warning("loan_status not present: cannot create is_default target automatically.")
        # if target already exists, leave as-is

    # date derived features (if present)
    if "issue_d" in df.columns:
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month
    if "earliest_cr_line" in df.columns:
        df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    if "last_pymnt_d" in df.columns and "issue_d" in df.columns:
        df["days_since_last_payment"] = (df["issue_d"] - df["last_pymnt_d"]).dt.days

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
    logger.info("Starting feature engineering pipeline.")
    df = load_data(input_path)
    logger.info(f"Data loaded: {df.shape}")

    df = create_target_and_basic_features(df)
    df_pca = apply_pca(df)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "engineered_data.parquet")
    df_pca.to_parquet(out_path, index=False)
    logger.info(f"Feature engineered data saved: {out_path}")
