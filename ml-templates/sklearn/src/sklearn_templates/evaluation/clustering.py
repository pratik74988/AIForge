from sklearn.metrics import silhouette_score


def evaluate_clustering(model, X):
    labels = model.predict(X) if hasattr(model, "predict") else model.labels_

    metrics = {
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
    }

    # Silhouette requires at least 2 clusters
    non_noise = labels != -1

    if len(set(labels[non_noise])) >= 2:
        metrics["silhouette_score"] = silhouette_score(
            X[non_noise],
            labels[non_noise],
        )

    return metrics