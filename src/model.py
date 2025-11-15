import os
import numpy as np
import pandas as pd
import warnings
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def load_processed_data(processed_dir: str):
    """Load train/test processed data"""
    X_train = pd.read_parquet(os.path.join(processed_dir, "X_train_processed.parquet"))
    X_test = pd.read_parquet(os.path.join(processed_dir, "X_test_processed.parquet"))
    y_train = pd.read_parquet(os.path.join(processed_dir, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(processed_dir, "y_test.parquet")).squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, name):
    """Evaluate a trained model"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    logger.info(f"{name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    return metrics


def tune_with_random_search(model, params, X_train, y_train, name):
    """Tune model using RandomizedSearchCV"""
    logger.info(f"Tuning {name}...")

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        cv=2,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    rs.fit(X_train, y_train)

    logger.info(f"{name} best params: {rs.best_params_}")
    return rs.best_estimator_


# Parameter grids

xgb_param_dist = {
    "n_estimators": [300, 500, 800, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2, 3]
}

lgbm_param_dist = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "num_leaves": [15, 31, 50],
    "max_depth": [3, 4, 5, -1],
    "min_child_samples": [100, 200, 300],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.3],
    "reg_lambda": [1, 2, 3]
}


cat_param_dist = {
    "iterations": [300, 500, 800],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05],
    "l2_leaf_reg": [1, 3, 5, 7],
    "border_count": [32, 64, 128]
}


def model_pipeline(processed_dir: str, model_output_dir: str):
    """Train, tune, ensemble, evaluate and save best model"""
    logger.info("Model pipeline started")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    # tuned_xgb = tune_with_random_search(
    #     XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42),
    #     xgb_param_dist, X_train, y_train, "XGBoost"
    # )

    tuned_lgbm = tune_with_random_search( 
        LGBMClassifier(random_state=42, force_col_wise=True,  min_split_gain=0.001,n_jobs=-1, verbose=-1),
        lgbm_param_dist, X_train, y_train, "LightGBM"
    )

    tuned_cat = tune_with_random_search(
        CatBoostClassifier(verbose=0, random_state=42),
        cat_param_dist, X_train, y_train, "CatBoost"
    )

    logger.info("Training stacking ensemble")

    stack = StackingClassifier(
        estimators=[
            # ("xgb", tuned_xgb),
            ("lgbm", tuned_lgbm),
            ("cat", tuned_cat)
        ],
        final_estimator=LogisticRegression(max_iter=500),
        stack_method="auto",
        n_jobs=-1
    )
    stack.fit(X_train, y_train)

    results = []
    # results.append(evaluate(tuned_xgb, X_test, y_test, "XGBoost"))
    results.append(evaluate(tuned_lgbm, X_test, y_test, "LightGBM"))
    results.append(evaluate(tuned_cat, X_test, y_test, "CatBoost"))
    results.append(evaluate(stack, X_test, y_test, "StackingEnsemble"))

    results_df = pd.DataFrame(results)

    best_row = results_df.iloc[results_df["roc_auc"].idxmax()]
    best_model_name = best_row["model"]

    best_model = {
        # "XGBoost": tuned_xgb,
        "LightGBM": tuned_lgbm,
        "CatBoost": tuned_cat,
        "StackingEnsemble": stack
    }[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_output_dir, "best_model.pkl"))
    results_df.to_csv(os.path.join(model_output_dir, "model_comparison.csv"), index=False)

    preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred": best_model.predict(X_test),
        "y_prob": best_model.predict_proba(X_test)[:, 1]
    })
    preds.to_parquet(os.path.join(model_output_dir, "predictions.parquet"), index=False)

    logger.info("Model pipeline completed")
