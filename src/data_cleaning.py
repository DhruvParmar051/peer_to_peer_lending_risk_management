import os
import numpy as np
import pandas as pd
import warnings

from utils.logger import get_logger
from utils.data_load import load_data
from utils.hybrid_iqr_capping import hybrid_iqr_capping, evaluate_capping_effect

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "Unnamed: 0", "id", "url", "title", "zip_code",
        "policy_code", "application_type", "hardship_flag",
        "hardship_type", "hardship_reason", "hardship_status",
        "payment_plan_start_date"
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped {len(existing)} unnecessary columns.")
    return df

def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing data types.")

    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    date_cols = [
        "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",
        "last_credit_pull_d", "sec_app_earliest_cr_line",
        "hardship_start_date", "hardship_end_date", "DATE"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Before handling missing values: {df.shape}")
    thresh = len(df) * 0.2
    df = df.dropna(thresh=thresh, axis=1)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode_val)

    logger.info(f"After handling missing values: {df.shape}")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    logger.info(f"Removed {before - df.shape[0]} duplicate rows.")
    return df

def clean_data_pipeline(input_path, output_dir):
    logger.info("Starting data cleaning pipeline.")

    # Load dataset
    df = load_data(input_path)

    # Drop unnecessary columns
    drop_cols = [
        "id", "member_id", "url", "desc", "zip_code", "emp_title",
        "title", "policy_code", "application_type", "pymnt_plan",
        "next_pymnt_d", "hardship_flag"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    logger.info(f"Dropped {len(drop_cols)} unnecessary columns.")

    # Standardize data types
    logger.info("Standardizing data types.")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
        elif np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  
        .str.replace('_+', '_', regex=True)            
        .str.strip('_')                                  
    )
    
    # Remove duplicates and handle missing values
    before_duplicates = df.shape[0]
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before_duplicates - df.shape[0]} duplicate rows.")
    logger.info(f"Before handling missing values: {df.shape}")

    threshold = 0.5
    df.dropna(axis=1, thresh=int((1 - threshold) * len(df)), inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    logger.info(f"After handling missing values: {df.shape}")

    # Hybrid IQR Outlier Treatment
    logger.info("Starting Hybrid IQR Outlier Treatment")
    df_capped, summary_df = hybrid_iqr_capping(df, cols=num_cols, factor=1.5)
    logger.info("Hybrid IQR capping completed.")

    # Create output directories
    output_files_dir = os.path.join(output_dir)
    os.makedirs(output_files_dir, exist_ok=True)

    # Save initial capping summary
    summary_path = os.path.join(output_files_dir, "hybrid_capping_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved capping summary: {summary_path}")

    # Analyze and drop heavily capped columns
    logger.info("Analyzing heavily capped columns for removal.")
    summary_df["mean_shift_ratio"] = (summary_df["mean_after"] - summary_df["mean_before"]).abs() / summary_df["mean_before"].replace(0, 1e-6)
    summary_df["std_reduction_ratio"] = (summary_df["std_before"] - summary_df["std_after"]).abs() / summary_df["std_before"].replace(0, 1e-6)

    MEAN_SHIFT_THRESHOLD = 0.25
    STD_REDUCTION_THRESHOLD = 0.50

    heavy_mean_shift = summary_df.loc[summary_df["mean_shift_ratio"] > MEAN_SHIFT_THRESHOLD, "column"]
    heavy_std_reduction = summary_df.loc[summary_df["std_reduction_ratio"] > STD_REDUCTION_THRESHOLD, "column"]
    drop_cols = sorted(set(heavy_mean_shift).union(set(heavy_std_reduction)))

    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} heavily capped columns: {drop_cols}")
        df_capped.drop(columns=[c for c in drop_cols if c in df_capped.columns], inplace=True)
    else:
        logger.info("No columns met heavy-capping thresholds.")

    # Save updated summary and cleaned data
    summary_path_final = os.path.join(output_files_dir, "hybrid_capping_summary_final.csv")
    summary_df.to_csv(summary_path_final, index=False)
    logger.info(f"Updated capping summary saved: {summary_path_final}")

    output_path = os.path.join(output_files_dir, "cleaned_data.parquet")
    df_capped.to_parquet(output_path, index=False)
    logger.info(f"Final cleaned data saved: {output_path}")
