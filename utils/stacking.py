# utils/stacking.py
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

def build_stacking_features(xgb, lgbm, cat, X):
    logger.info("Building stacking features for data of shape %s", X.shape)
    base = pd.DataFrame({
        "xgb": xgb.predict_proba(X)[:, 1],
        "lgbm": lgbm.predict_proba(X)[:, 1],
        "cat": cat.predict_proba(X)[:, 1]
    })

    stack = base.copy()
    stack["min"] = base.min(axis=1)
    stack["max"] = base.max(axis=1)
    stack["mean"] = base.mean(axis=1)
    stack["std"] = base.std(axis=1)
    stack["xgb_lgbm"] = base["xgb"] * base["lgbm"]
    stack["xgb_cat"] = base["xgb"] * base["cat"]
    stack["lgbm_cat"] = base["lgbm"] * base["cat"]
    stack["risk_signal"] = 0.6 * base["xgb"] + 0.3 * base["lgbm"] + 0.1 * base["cat"]

    return stack

def tune_threshold_f2(probs, y_true, n_steps=300):
    logger.info("Tuning threshold using F2 with %d steps", n_steps)
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t = 0.5
    best_f2 = -1.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        denom = (4 * precision + recall + 1e-9)
        f2 = (5 * precision * recall) / denom if denom > 0 else 0.0

        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    logger.info("Tuned threshold = %.4f with F2 = %.4f", best_t, best_f2)
    return best_t, best_f2

def train_meta_model(xgb, lgbm, cat, X_val, y_val):
    logger.info("Training meta-model on validation set of shape %s", X_val.shape)
    val_features = build_stacking_features(xgb, lgbm, cat, X_val)

    meta_model = XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=5
    )

    meta_model.fit(val_features, y_val)
    val_probs = meta_model.predict_proba(val_features)[:, 1]
    best_threshold, best_f2 = tune_threshold_f2(val_probs, y_val)
    logger.info("Meta-model trained. Best threshold: %.4f", best_threshold)
    return meta_model, best_threshold

def predict_meta(meta_model, xgb, lgbm, cat, X_test, threshold):
    test_features = build_stacking_features(xgb, lgbm, cat, X_test)
    probs = meta_model.predict_proba(test_features)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs
