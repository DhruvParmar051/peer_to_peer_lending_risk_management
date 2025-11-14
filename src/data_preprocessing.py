import os
import numpy as np
import pandas as pd
import logging
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.data_load import load_data

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, target_col: str = "is_default"):
    """
    Split the dataset into train and test sets.
    Should be done before fitting preprocessing steps.
    """
    logger.info("Splitting data into train and test sets.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Preprocess data by handling missing values, scaling numeric columns,
    and encoding categorical columns.
    """
    logger.info("Starting preprocessing of features.")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    low_cat_cols = [c for c in categorical_cols if X_train[c].nunique() < 10]
    high_cat_cols = [c for c in categorical_cols if X_train[c].nunique() >= 10]

    low_cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    high_cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("low_cat", low_cat_pipeline, low_cat_cols),
            ("high_cat", high_cat_pipeline, high_cat_cols)
        ],
        remainder="drop"
    )

    # Fit on train, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    low_cat_features = (
        preprocessor.named_transformers_["low_cat"]["onehot"].get_feature_names_out(low_cat_cols)
        if low_cat_cols else []
    )
    high_cat_features = high_cat_cols
    all_features = list(numeric_cols) + list(low_cat_features) + list(high_cat_features)

    X_train_df = pd.DataFrame(X_train_processed, columns=all_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_features, index=X_test.index)

    logger.info("Preprocessing complete.")
    return X_train_df, X_test_df


def data_preprocessing_pipeline(input_path: str, output_dir: str):
    """
    Full preprocessing pipeline:
    1. Load feature-engineered data
    2. Split into train/test
    3. Apply imputation, scaling, and encoding
    4. Save processed data
    """
    logger.info("Starting data preprocessing pipeline.")

    df = load_data(input_path)
    logger.info(f"Data loaded: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)

    os.makedirs(output_dir, exist_ok=True)

    X_train_processed.to_parquet(os.path.join(output_dir, "X_train_processed.parquet"), index=False)
    X_test_processed.to_parquet(os.path.join(output_dir, "X_test_processed.parquet"), index=False)
    y_train.to_frame(name="is_default").to_parquet(
    os.path.join(output_dir, "y_train.parquet"), index=False
    )
    y_test.to_frame(name="is_default").to_parquet(
    os.path.join(output_dir, "y_test.parquet"), index=False
    )

    logger.info(f"Preprocessed data saved to: {output_dir}")
    logger.info("Data preprocessing pipeline completed successfully.")