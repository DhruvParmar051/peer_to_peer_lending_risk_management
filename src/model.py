import os
import numpy as np
import pandas as pd
import warnings
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def load_processed_data(processed_dir: str):
    X_train = pd.read_parquet(os.path.join(processed_dir, "X_train_processed.parquet"))
    X_test = pd.read_parquet(os.path.join(processed_dir, "X_test_processed.parquet"))

    y_train = pd.read_parquet(os.path.join(processed_dir, "y_train.parquet"))["is_default"]
    y_test = pd.read_parquet(os.path.join(processed_dir, "y_test.parquet"))["is_default"]

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    logger.info(f"{name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    return metrics


def tune_with_random_search(model, params, X_train, y_train, name, model_output_dir):
    logger.info(f"Tuning {name}...")

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=15,
        cv=3,
        scoring="f1",
        n_jobs=4,
        verbose=2,
        random_state=42,
        refit=True
    )

    rs.fit(X_train, y_train)
    logger.info(f"{name} best params: {rs.best_params_}")

    os.makedirs(model_output_dir, exist_ok=True)
    
    joblib.dump(rs, os.path.join(model_output_dir, f"best_{name}.pkl"))
    
    return rs.best_estimator_


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 200)

    best_f1 = 0
    best_threshold = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1

xgb_param_dist = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 2, 3]
}

lgbm_param_dist = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "num_leaves": [15, 31, 50],
    "max_depth": [-1, 4, 6],
    "min_child_samples": [50, 100, 200],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.3],
    "reg_lambda": [1, 2, 3]
}

cat_param_dist = {
    "iterations": [300, 500, 800],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05],
    "l2_leaf_reg": [1, 3, 5]
}


def model_pipeline(processed_dir: str, model_output_dir: str):
    logger.info("Model pipeline started")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    # tuned_xgb = tune_with_random_search(
    #     XGBClassifier(
    #         objective="binary:logistic",
    #         eval_metric="logloss",
    #         tree_method="hist",
    #         max_bin=256,
    #         random_state=42,
    #         n_jobs=-1,
    #     ),
    #     xgb_param_dist, X_train, y_train, "XGBoost", model_output_dir
    # )

    # tuned_lgbm = tune_with_random_search(
    #     LGBMClassifier(
    #         random_state=42,
    #         n_jobs=-1,
    #         force_col_wise=True,
    #         verbose=-1
    #     ),
    #     lgbm_param_dist, X_train, y_train, "LightGBM", model_output_dir
    # )

    # tuned_cat = tune_with_random_search(
    #     CatBoostClassifier(
    #         verbose=0,
    #         random_state=42,
    #         thread_count=-1
    #     ),
    #     cat_param_dist, X_train, y_train, "CatBoost", model_output_dir
    # )
    
    tuned_xgb = joblib.load(r'D:\DAU\SEM 1\DS605 - Fundamentals of Machine Learning\peer_to_peer_lending_risk_management\models\best_XGBoost.pkl')
    tuned_lgbm = joblib.load(r'D:\DAU\SEM 1\DS605 - Fundamentals of Machine Learning\peer_to_peer_lending_risk_management\models\best_LightGBM.pkl')
    tuned_cat = joblib.load(r'D:\DAU\SEM 1\DS605 - Fundamentals of Machine Learning\peer_to_peer_lending_risk_management\models\best_CatBoost.pkl')
    
    results = []
    results.append(evaluate(tuned_xgb, X_test, y_test, "XGBoost"))
    results.append(evaluate(tuned_lgbm, X_test, y_test, "LightGBM"))
    results.append(evaluate(tuned_cat, X_test, y_test, "CatBoost"))

    results_df = pd.DataFrame(results)
    best_row = results_df.iloc[results_df["roc_auc"].idxmax()]
    best_model_name = best_row["model"]

    best_model = {
        "XGBoost": tuned_xgb,
        "LightGBM": tuned_lgbm,
        "CatBoost": tuned_cat,
    }[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    logger.info("Running threshold tuning...")

    best_model = joblib.load(r'D:\DAU\SEM 1\DS605 - Fundamentals of Machine Learning\peer_to_peer_lending_risk_management\models\best_model.pkl')

    y_prob = best_model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_test, y_prob)

    logger.info(f"Optimal Threshold: {best_threshold:.3f}")
    logger.info(f"Best F1 at this threshold: {best_f1:.4f}")

    y_pred_thresh = (y_prob >= best_threshold).astype(int)

    preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_thresh,
        "y_prob": y_prob,
        "threshold_used": best_threshold
    })

    os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_output_dir, "best_model.pkl"))
    results_df.to_csv(os.path.join(model_output_dir, "model_comparison.csv"), index=False)

    preds.to_parquet(os.path.join(model_output_dir, "predictions.parquet"), index=False)

    logger.info("Model pipeline completed")
