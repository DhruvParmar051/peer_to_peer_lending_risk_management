import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from utils.data_load import load_data
from utils.logger import get_logger
from utils.hybrid_iqr_capping import hybrid_iqr_capping

logger = get_logger(__name__)


def split_data(df: pd.DataFrame, target_col="is_default"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test



def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = [c for c in categorical_cols if X_train[c].nunique() < 10]
    high_card = [c for c in categorical_cols if X_train[c].nunique() >= 10]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("low_cat", low_cat_pipe, low_card),
        ("high_cat", high_cat_pipe, high_card),
    ])

    return preprocessor, numeric_cols, low_card, high_card


def data_preprocessing_pipeline(input_path, output_dir):
    df = load_data(input_path)

    X_train, X_test, y_train, y_test = split_data(df)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if "is_default" in num_cols:
        num_cols.remove("is_default")

    X_train_capped, caps = hybrid_iqr_capping(X_train.copy(), num_cols)


    # Apply same caps to test set
    X_test_capped = X_test.copy()
    
    for col, bound in caps.items():
        if col in X_test_capped.columns:
            X_test_capped[col] = X_test_capped[col].clip(bound["lower"], bound["upper"])

    X_train = X_train_capped
    X_test = X_test_capped

    preprocessor, num_cols, low_cat_cols, high_cat_cols = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    if low_cat_cols:
        ohe = preprocessor.named_transformers_["low_cat"]["onehot"]
        low_cat_features = list(ohe.get_feature_names_out(low_cat_cols))
    else:
        low_cat_features = []

    feature_names = num_cols + low_cat_features + high_cat_cols

    X_train_df = pd.DataFrame(X_train_p, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)

    os.makedirs(output_dir, exist_ok=True)
    X_train_df.to_parquet(os.path.join(output_dir, "X_train_processed.parquet"), index=False)
    X_test_df.to_parquet(os.path.join(output_dir, "X_test_processed.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(output_dir, "y_test.parquet"), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
    joblib.dump(caps, os.path.join(output_dir, "hybrid_iqr_caps.pkl"))
