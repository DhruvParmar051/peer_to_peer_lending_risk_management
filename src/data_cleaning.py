import os
import numpy as np
import pandas as pd
from utils.data_load import load_data
from utils.logger import get_logger
import re

logger = get_logger(__name__)


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
            "last_credit_pull_d", "sec_app_earliest_cr_line",
            "hardship_start_date", "hardship_end_date", "DATE"
        ]
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    return df


def drop_unwanted_columns(df: pd.DataFrame, drop_list=None) -> pd.DataFrame:
    if drop_list is None:
        drop_list = [
            "id", "member_id", "url", "desc", "zip_code", "emp_title",
            "policy_code", "application_type", "pymnt_plan", "hardship_flag", "out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee","last_pymnt_amnt","last_pymnt_d","last_credit_pull_d","days_since_last_payment","issue_d", 'Unnamed_0'
        ]
    existing = [c for c in drop_list if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logger.info(f"Dropped {len(existing)} columns: {existing}")
    return df


def handle_missing_values(df: pd.DataFrame, col_threshold=0.5) -> pd.DataFrame:
    min_non_na = int(col_threshold * len(df))
    df = df.dropna(axis=1, thresh=min_non_na)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        if df[c].isnull().any():
            mode = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode)

    return df


def clean_data_pipeline(input_path: str, output_dir: str):
    logger.info("Starting data cleaning pipeline.")
    df = load_data(input_path)

    df = sanitize_column_names(df)
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    df = convert_percentage_columns(df)
    df = parse_date_columns(df)
    df = drop_unwanted_columns(df)
    df = handle_missing_values(df)

    os.makedirs(output_dir, exist_ok=True)
    cleaned_path = os.path.join(output_dir, "cleaned_data.parquet")
    df.to_parquet(cleaned_path, index=False)

    logger.info(f"Cleaned data saved to {cleaned_path}")
    logger.info("Data cleaning pipeline completed.")
