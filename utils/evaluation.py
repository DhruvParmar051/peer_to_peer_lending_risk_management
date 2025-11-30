from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test, name):
    logger.info("Evaluating model: %s", name)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    logger.info("%s metrics: AUC=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                name, metrics["roc_auc"], metrics["f1"], metrics["precision"], metrics["recall"])
    return metrics
