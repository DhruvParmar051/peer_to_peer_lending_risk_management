# model.py
import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from utils.logger import get_logger
from utils.stacking import build_stacking_features, train_meta_model, predict_meta
from utils.tuning import tune_with_random_search
from utils.evaluation import evaluate

warnings.filterwarnings("ignore")
logger = get_logger("model_pipeline")

def load_processed_data(processed_dir: str):
    X_train = pd.read_parquet(os.path.join(processed_dir, "X_train_processed.parquet"))
    X_test = pd.read_parquet(os.path.join(processed_dir, "X_test_processed.parquet"))
    y_train = pd.read_parquet(os.path.join(processed_dir, "y_train.parquet"))["is_default"]
    y_test = pd.read_parquet(os.path.join(processed_dir, "y_test.parquet"))["is_default"]
    return X_train, X_test, y_train, y_test

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
    "num_leaves": [31, 50, 64],
    "max_depth": [-1, 4, 6],
    "min_child_samples": [50, 100],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 2]
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

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    pos_weight = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    logger.info("Using scale_pos_weight=%.3f", pos_weight)

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight
    )

    lgbm_model = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
        is_unbalance=True
    )

    cat_model = CatBoostClassifier(
        verbose=0,
        random_state=42,
        thread_count=-1,
        class_weights=[1, pos_weight]
    )

    tuned_xgb = tune_with_random_search(xgb_model, xgb_param_dist, X_tr, y_tr, "XGBoost", model_output_dir, n_iter=10, cv=3)
    tuned_lgbm = tune_with_random_search(lgbm_model, lgbm_param_dist, X_tr, y_tr, "LightGBM", model_output_dir, n_iter=10, cv=3)
    tuned_cat = tune_with_random_search(cat_model, cat_param_dist, X_tr, y_tr, "CatBoost", model_output_dir, n_iter=10, cv=3)

    try:
        tuned_xgb = joblib.load(os.path.join(os.getcwd(), "models", "best_XGBoost.pkl"))
        tuned_lgbm = joblib.load(os.path.join(os.getcwd(), "models", "best_LightGBM.pkl"))
        tuned_cat = joblib.load(os.path.join(os.getcwd(), "models", "best_CatBoost.pkl"))
        logger.info("Loaded pretrained tuned models from ./models/")
    except Exception:
        logger.info("No pretrained tuned models found in ./models/ — using tuned estimators from search")

    balanced_xgb = BalancedBaggingClassifier(
        estimator=tuned_xgb,
        sampling_strategy="auto",
        n_estimators=2,
        replacement=False,
        n_jobs=-1,
        random_state=42
    )
    balanced_xgb.fit(X_train, y_train)
    logger.info("Balanced bagging fitted on full training data")

    meta_model, best_threshold = train_meta_model(balanced_xgb, tuned_lgbm, tuned_cat, X_val, y_val)

    y_test_pred, test_probs = predict_meta(meta_model, balanced_xgb, tuned_lgbm, tuned_cat, X_test, best_threshold)

    final_metrics = evaluate(meta_model, build_stacking_features(balanced_xgb, tuned_lgbm, tuned_cat, X_test), y_test, "Stacked_Meta_Model")
    final_metrics["threshold"] = best_threshold
    final_metrics["y_pred"] = y_test_pred

    logger.info("Final test metrics: %s", {k: final_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]})

    os.makedirs(model_output_dir, exist_ok=True)
    joblib.dump({"xgb": tuned_xgb, "lgbm": tuned_lgbm, "cat": tuned_cat, "meta": meta_model},
                os.path.join(model_output_dir, "stacked_models.pkl"))

    pd.DataFrame([final_metrics]).to_csv(os.path.join(model_output_dir, "final_results.csv"), index=False)

    pd.DataFrame({
        "y_true": y_test,
        "y_prob": test_probs,
        "y_pred": final_metrics["y_pred"],
        "threshold": best_threshold
    }).to_parquet(os.path.join(model_output_dir, "predictions.parquet"), index=False)

    logger.info("Model pipeline completed successfully.")
    return final_metrics
