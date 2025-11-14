import os
import pandas as pd
import logging
import warnings
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from utils.data_load import load_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data(processed_dir: str):
    """
    Load preprocessed train/test data.
    """
    logger.info("Loading preprocessed data for model tuning.")
    X_train = pd.read_parquet(os.path.join(processed_dir, "X_train_processed.parquet"))
    X_test = pd.read_parquet(os.path.join(processed_dir, "X_test_processed.parquet"))
    y_train = pd.read_parquet(os.path.join(processed_dir, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(processed_dir, "y_test.parquet")).squeeze()

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for XGBoost.
    """
    logger.info("Starting XGBoost hyperparameter tuning.")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    param_dist = {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "gamma": [0.1, 0.2]
    }

    rs = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=2,
    random_state=42
    )


    rs.fit(X_train, y_train)

    logger.info(f"Best Parameters: {rs.best_params_}")
    logger.info(f"Best F1 Score (CV): {rs.best_score_:.4f}")

    best_model = rs.best_estimator_
    return best_model, rs.best_params_, rs.best_score_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate tuned XGBoost model on test data.
    """
    logger.info("Evaluating tuned XGBoost model on test data.")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    logger.info("Evaluation Results:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

    return metrics


def model_tuning_pipeline(processed_dir: str, model_output_dir: str):
    """
    Complete pipeline for XGBoost model tuning:
    1. Load processed data
    2. Tune hyperparameters
    3. Evaluate tuned model
    4. Save tuned model and results
    """
    logger.info("Starting XGBoost model tuning pipeline.")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    tuned_model, best_params, best_cv_f1 = tune_xgboost(X_train, y_train)
    metrics = evaluate_model(tuned_model, X_test, y_test)

    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "xgboost_tuned.pkl")
    joblib.dump(tuned_model, model_path)

    logger.info(f"Tuned XGBoost model saved at: {model_path}")

    # Save metrics and parameters
    results = {**metrics, **best_params, "cv_f1": best_cv_f1}
    results_path = os.path.join(model_output_dir, "xgboost_tuning_results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)

    logger.info(f"Model tuning results saved to: {results_path}")
    logger.info("XGBoost model tuning pipeline completed successfully.")


