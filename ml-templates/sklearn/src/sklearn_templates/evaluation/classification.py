from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)

            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_proba[:, 1]
                )
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="weighted",
                )
        except ValueError:
            pass

    return metrics