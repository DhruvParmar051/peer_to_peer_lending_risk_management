# src/data_cleaning.py
import os
import numpy as np
import pandas as pd
from utils.data_load import load_data
from utils.logger import get_logger
from utils.hybrid_iqr_capping import hybrid_iqr_capping  # assumes exists and works
import re

logger = get_logger(__name__)

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # replace special chars with underscore, collapse multiple underscores, strip edges
    cols = (
        df.columns
        .astype(str)
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.replace('_+', '_', regex=True)
        .str.strip('_')
    )
    df.columns = cols
    return df

def convert_percentage_columns(df: pd.DataFrame, candidates=None) -> pd.DataFrame:
    if candidates is None:
        candidates = ["int_rate", "revol_util"]
    for col in candidates:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).replace({'nan': np.nan})
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def parse_date_columns(df: pd.DataFrame, candidates=None) -> pd.DataFrame:
    if candidates is None:
        candidates = [
            "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",
            "last_credit_pull_d", "sec_app_earliest_cr_line", "hardship_start_date",
            "hardship_end_date", "DATE"
        ]
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    return df

def drop_unwanted_columns(df: pd.DataFrame, drop_list=None) -> pd.DataFrame:
    if drop_list is None:
        drop_list = [
            "id", "member_id", "url", "desc", "zip_code", "emp_title",
            "policy_code", "application_type", "pymnt_plan", "hardship_flag"
        ]
    existing = [c for c in drop_list if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logger.info(f"Dropped {len(existing)} columns: {existing}")
    return df

def handle_missing_values(df: pd.DataFrame, col_threshold=0.5) -> pd.DataFrame:
    # drop columns with more than (1-col_threshold) missing ratio
    min_non_na = int(col_threshold * len(df))
    df = df.dropna(axis=1, thresh=min_non_na)
    # numeric median fill, categorical mode fill
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        if df[c].isnull().any():
            mode = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode)
    return df

def clean_data_pipeline(input_path: str, output_dir: str):
    """
    Standardized cleaning pipeline:
    - Load data
    - Sanitize column names
    - Convert percent fields
    - Parse dates
    - Drop obvious unwanted columns
    - Remove duplicates and fill missing
    - Apply hybrid IQR capping to numeric features
    - Save cleaned parquet
    """
    logger.info("Starting data cleaning pipeline.")
    df = load_data(input_path)

    # sanitize
    df = sanitize_column_names(df)

    # drop duplicates early
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    df = convert_percentage_columns(df)
    df = parse_date_columns(df)

    df = drop_unwanted_columns(df)

    df = handle_missing_values(df)

    # hybrid capping for numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        logger.info("Applying hybrid IQR capping on numeric columns.")
        df_capped, summary_df = hybrid_iqr_capping(df, cols=num_cols, factor=1.5)
    else:
        df_capped = df
        summary_df = pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    cleaned_path = os.path.join(output_dir, "cleaned_data.parquet")
    df_capped.to_parquet(cleaned_path, index=False)
    logger.info(f"Cleaned data saved to {cleaned_path}")

    # save summary if available
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, "hybrid_capping_summary.csv"), index=False)
    logger.info("Data cleaning pipeline completed.")
