import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
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
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
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
        n_jobs=6,
        verbose=2,
        random_state=42
    )

    rs.fit(X_train, y_train)
    logger.info(f"{name} best params: {rs.best_params_}")

    os.makedirs(model_output_dir, exist_ok=True)
    joblib.dump(rs, os.path.join(model_output_dir, f"best_{name}.pkl"))

    return rs.best_estimator_


# parameter grids as before
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

    # train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
    logger.info(f"Using scale_pos_weight={pos_weight:.3f}")

    # base models
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

    # using tuned models
    tuned_xgb = tune_with_random_search(xgb_model, xgb_param_dist, X_tr, y_tr, "XGBoost", model_output_dir)
    tuned_lgbm = tune_with_random_search(lgbm_model, lgbm_param_dist, X_tr, y_tr, "LightGBM", model_output_dir)
    tuned_cat = tune_with_random_search(cat_model, cat_param_dist, X_tr, y_tr, "CatBoost", model_output_dir)

    # stacking validation features (base probabilities)
    val_base = pd.DataFrame({
        "xgb": tuned_xgb.predict_proba(X_val)[:, 1],
        "lgbm": tuned_lgbm.predict_proba(X_val)[:, 1],
        "cat": tuned_cat.predict_proba(X_val)[:, 1]
    })

    # full stacking features
    val_stack_X = val_base.copy()
    val_stack_X["min"] = val_base.min(axis=1)
    val_stack_X["max"] = val_base.max(axis=1)
    val_stack_X["mean"] = val_base.mean(axis=1)
    val_stack_X["std"] = val_base.std(axis=1)

    val_stack_X["xgb_lgbm"] = val_base["xgb"] * val_base["lgbm"]
    val_stack_X["xgb_cat"] = val_base["xgb"] * val_base["cat"]
    val_stack_X["lgbm_cat"] = val_base["lgbm"] * val_base["cat"]


    logger.info("stacking with full feature set and XGBoost meta-model")

    # XGBoost meta-model
    meta_model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    meta_model.fit(val_stack_X, y_val)

    ensemble_val = meta_model.predict_proba(val_stack_X)[:, 1]

    # precision-focused threshold tuning using F0.5
    thresholds = np.linspace(0.01, 0.99, 300)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        pred = (ensemble_val >= t).astype(int)
        precision = precision_score(y_val, pred)
        recall = recall_score(y_val, pred)

        f1 = (2 * precision * recall) / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    logger.info(f"Selected precision-tuned threshold (F1): {best_threshold:.4f}")

    # test stacking with full features
    test_base = pd.DataFrame({
        "xgb": tuned_xgb.predict_proba(X_test)[:, 1],
        "lgbm": tuned_lgbm.predict_proba(X_test)[:, 1],
        "cat": tuned_cat.predict_proba(X_test)[:, 1]
    })
    
    test_stack_X = test_base.copy()
    test_stack_X["min"] = test_base.min(axis=1)
    test_stack_X["max"] = test_base.max(axis=1)
    test_stack_X["mean"] = test_base.mean(axis=1)
    test_stack_X["std"] = test_base.std(axis=1)
    
    test_stack_X["xgb_lgbm"] = test_base["xgb"] * test_base["lgbm"]
    test_stack_X["xgb_cat"] = test_base["xgb"] * test_base["cat"]
    test_stack_X["lgbm_cat"] = test_base["lgbm"] * test_base["cat"]

    test_probs = meta_model.predict_proba(test_stack_X)[:, 1]
    
    y_test_pred = (test_probs >= best_threshold).astype(int)

    final_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "classification_report": classification_report(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "threshold": best_threshold
    }

    logger.info("Final test metrics:")
    logger.info(final_metrics)

    os.makedirs(model_output_dir, exist_ok=True)
    joblib.dump(
        {"xgb": tuned_xgb, "lgbm": tuned_lgbm, "cat": tuned_cat, "meta": meta_model},
        os.path.join(model_output_dir, "stacked_models.pkl")
    )

    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(model_output_dir, "final_results.csv"), index=False
    )

    pd.DataFrame({
        "y_true": y_test,
        "y_prob": test_probs,
        "y_pred": y_test_pred,
        'thresholds': best_threshold
    }).to_parquet(os.path.join(model_output_dir, "predictions.parquet"), index=False)

    logger.info("Model pipeline completed successfully.")

    return final_metrics
