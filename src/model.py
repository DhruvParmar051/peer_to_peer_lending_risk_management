import os
import pandas as pd
import numpy as np
import logging
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data(processed_dir: str):
    """
    Load preprocessed training and test data.
    """
    logger.info("Loading preprocessed data.")

    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_processed.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test_processed.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Logistic Regression model on the training data.
    """
    logger.info("Training Logistic Regression model.")

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    logger.info("Model training completed successfully.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the trained model on the test data and print metrics.
    """
    logger.info("Evaluating model on test data.")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    logger.info("Model Evaluation Metrics:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")

    logger.info("\nDetailed Classification Report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")


def save_model(model, output_dir: str):
    """
    Save the trained model as a .pkl file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Trained model saved to: {model_path}")

def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=42
        )
    }

    results = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        logger.info(f"{name} Results:")
        logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc
        })

    results_df = pd.DataFrame(results)
    logger.info("\nModel Comparison:\n" + str(results_df))
    return results_df


def model_pipeline(processed_dir: str, model_output_dir: str):
    """
    Full model pipeline:
    1. Load processed data
    2. Train Logistic Regression
    3. Evaluate model
    4. Save trained model
    """
    logger.info("Starting model training pipeline.")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    compare_models(X_train, y_train, X_test, y_test)
    logger.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    model_output_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_output_dir, exist_ok=True)

    model_pipeline(processed_dir, model_output_dir)
