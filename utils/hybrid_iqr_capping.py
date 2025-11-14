import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def hybrid_iqr_capping(df: pd.DataFrame, cols: list, factor: float = 1.5):
    """
    Caps extreme values using IQR bounds instead of removing rows.
    Returns both the capped DataFrame and a capping summary.
    """
    logging.info(f"Columns: {len(cols)} | Factor: {factor}")

    df_before = df.copy()
    for col in cols:
        if col not in df.columns or df[col].dtype.kind not in "bifc":
            logging.warning(f"Skipping non-numeric or missing column: {col}")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        capped = np.clip(df[col], lower, upper)
        num_capped = (df[col] != capped).sum()
        df[col] = capped

        logging.info(f"{col}: capped {num_capped} values [{lower:.2f}, {upper:.2f}]")

    summary_df = evaluate_capping_effect(df_before, df, cols)
    logging.info("Hybrid IQR capping completed (no rows removed).")
    return df, summary_df


def evaluate_capping_effect(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: list):
    """
    Compare distributions before and after capping and return a summary DataFrame.
    """
    summary = []
    for col in cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        stats_before = df_before[col].describe()
        stats_after = df_after[col].describe()
        summary.append({
            "column": col,
            "mean_before": stats_before["mean"],
            "mean_after": stats_after["mean"],
            "std_before": stats_before["std"],
            "std_after": stats_after["std"],
            "min_before": stats_before["min"],
            "min_after": stats_after["min"],
            "max_before": stats_before["max"],
            "max_after": stats_after["max"]
        })
    
    summary_df = pd.DataFrame(summary)
    logging.info("Capping evaluation summary generated.")
    return summary_df
