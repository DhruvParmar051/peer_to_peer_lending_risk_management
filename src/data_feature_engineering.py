import os
import numpy as np
import pandas as pd
from utils.data_load import load_data
from utils.logger import get_logger

logger = get_logger(__name__)


def create_target_and_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating target variable 'is_default'.")

    valid_status = ["Fully Paid", "Charged Off", "Default"]
    if "loan_status" in df.columns:
        df = df[df["loan_status"].isin(valid_status)].copy()
        df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
        df = df.drop(columns=["loan_status"])

    if "issue_d" in df.columns:
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
        df = df.drop(columns=["earliest_cr_line"], errors="ignore")

    return df


def feature_engineering_pipeline(input_path: str, output_dir: str):
    logger.info("Starting feature engineering pipeline.")
    df = load_data(input_path)
    logger.info(f"Data loaded: {df.shape}")

    df = create_target_and_basic_features(df)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "engineered_data.parquet")
    df.to_parquet(out_path, index=False)

    logger.info(f"Feature engineered data saved: {out_path}")
