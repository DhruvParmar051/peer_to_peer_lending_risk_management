# src/data_preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.data_load import load_data
from utils.logger import get_logger

logger = get_logger(__name__)

def split_data(df: pd.DataFrame, target_col: str = "is_default"):
    logger.info("Splitting data into train and test sets.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = [c for c in categorical_cols if X_train[c].nunique() < 10]
    high_card = [c for c in categorical_cols if X_train[c].nunique() >= 10]

    logger.info(f"Numeric: {len(numeric_cols)}, Low-card-cat: {len(low_card)}, High-card-cat: {len(high_card)}")

    # pipelines
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("low_cat", low_cat_pipe, low_card),
        ("high_cat", high_cat_pipe, high_card),
    ], remainder="drop")

    feature_names = {
        "numeric": numeric_cols,
        "low_cat": low_card,
        "high_cat": high_card
    }

    return preprocessor, feature_names

def data_preprocessing_pipeline(input_path: str, output_dir: str):
    logger.info("Starting preprocessing pipeline.")
    df = load_data(input_path)
    logger.info(f"Loaded for preprocessing: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor, feature_names = build_preprocessor(X_train)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # build column names
    low_cat_cols = feature_names["low_cat"]
    high_cat_cols = feature_names["high_cat"]
    numeric_cols = feature_names["numeric"]

    # onehot feature names
    if low_cat_cols:
        ohe = preprocessor.named_transformers_["low_cat"]["onehot"]
        low_cat_features = list(ohe.get_feature_names_out(low_cat_cols))
    else:
        low_cat_features = []

    all_features = list(numeric_cols) + low_cat_features + list(high_cat_cols)

    X_train_df = pd.DataFrame(X_train_proc, columns=all_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=all_features, index=X_test.index)

    os.makedirs(output_dir, exist_ok=True)
    X_train_df.to_parquet(os.path.join(output_dir, "X_train_processed.parquet"), index=False)
    X_test_df.to_parquet(os.path.join(output_dir, "X_test_processed.parquet"), index=False)
    # save y as dataframe to keep to_parquet method consistent
    y_train.to_frame(name="is_default").to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False)
    y_test.to_frame(name="is_default").to_parquet(os.path.join(output_dir, "y_test.parquet"), index=False)

    # persist preprocessor for inference
    import joblib
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))

    logger.info(f"Preprocessing completed and saved to: {output_dir}")
