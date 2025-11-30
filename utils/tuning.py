import os
import joblib
from sklearn.model_selection import RandomizedSearchCV
from utils.logger import get_logger

logger = get_logger(__name__)

def tune_with_random_search(model, params, X_train, y_train, name, model_output_dir, n_iter=10, cv=3):
    logger.info("Starting RandomizedSearchCV for %s", name)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    rs.fit(X_train, y_train)
    logger.info("%s best params: %s", name, rs.best_params_)

    os.makedirs(model_output_dir, exist_ok=True)
    out_path = os.path.join(model_output_dir, f"best_{name}.pkl")
    joblib.dump(rs, out_path)
    logger.info("Saved RandomizedSearchCV result to %s", out_path)

    return rs.best_estimator_
