import numpy as np
import pandas as pd
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
>>>>>>> 81b372d (data cleaning created)
=======

from utils.logger import get_logger   
 
logger = get_logger(__name__)

>>>>>>> 9a1af4c (created new logger module)

def hybrid_iqr_capping(df: pd.DataFrame, cols: list, factor: float = 1.5):
    """
    Caps extreme values using IQR bounds instead of removing rows.
    Returns both the capped DataFrame and a capping summary.
    """
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    logger.info(f"Columns: {len(cols)} | Factor: {factor}")
=======
    logging.info("Starting Hybrid IQR Outlier Treatment")
=======
>>>>>>> 91a139b (pipeline updated)
    logging.info(f"Columns: {len(cols)} | Factor: {factor}")
>>>>>>> 81b372d (data cleaning created)
=======
    logger.info(f"Columns: {len(cols)} | Factor: {factor}")
>>>>>>> 9a1af4c (created new logger module)

    df_before = df.copy()
    for col in cols:
        if col not in df.columns or df[col].dtype.kind not in "bifc":
<<<<<<< HEAD
<<<<<<< HEAD
            logger.warning(f"Skipping non-numeric or missing column: {col}")
=======
            logging.warning(f"Skipping non-numeric or missing column: {col}")
>>>>>>> 81b372d (data cleaning created)
=======
            logger.warning(f"Skipping non-numeric or missing column: {col}")
>>>>>>> 9a1af4c (created new logger module)
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        capped = np.clip(df[col], lower, upper)
        num_capped = (df[col] != capped).sum()
        df[col] = capped

<<<<<<< HEAD
<<<<<<< HEAD
        logger.info(f"{col}: capped {num_capped} values [{lower:.2f}, {upper:.2f}]")

    summary_df = evaluate_capping_effect(df_before, df, cols)
    logger.info("Hybrid IQR capping completed (no rows removed).")
=======
        logging.info(f"{col}: capped {num_capped} values [{lower:.2f}, {upper:.2f}]")

    summary_df = evaluate_capping_effect(df_before, df, cols)
    logging.info("Hybrid IQR capping completed (no rows removed).")
>>>>>>> 81b372d (data cleaning created)
=======
        logger.info(f"{col}: capped {num_capped} values [{lower:.2f}, {upper:.2f}]")

    summary_df = evaluate_capping_effect(df_before, df, cols)
    logger.info("Hybrid IQR capping completed (no rows removed).")
>>>>>>> 9a1af4c (created new logger module)
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
<<<<<<< HEAD
<<<<<<< HEAD
    logger.info("Capping evaluation summary generated.")
=======
    logging.info("Capping evaluation summary generated.")
>>>>>>> 81b372d (data cleaning created)
=======
    logger.info("Capping evaluation summary generated.")
>>>>>>> 9a1af4c (created new logger module)
    return summary_df
