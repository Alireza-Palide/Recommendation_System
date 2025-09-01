from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from src.config.config_loader import get_config  # type: ignore
from src.utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)
config = get_config()


MODEL_PATH = Path("models/kmeans_model.pkl")


def get_clustering_model(method: str, n_clusters: int):

    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "minibatchkmeans":
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def find_optimal_k(data: pd.DataFrame, max_k: int = 10):

    logger.info("Searching for optimal number of clusters using silhouette score...")
    best_k = 2
    best_score = -1

    for k in range(3, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        logger.info(f"Silhouette score for k={k}: {score:.4f}")

        if score > best_score:
            best_k = k
            best_score = score

    logger.info(f"Best number of clusters found: {best_k} with score {best_score:.4f}")
    return best_k


def cluster_rfm_data(data: pd.DataFrame) -> pd.DataFrame:

    method = config["clustering"].get("method", "kmeans")
    max_clusters = config["clustering"].get("max_clusters", 10)

    optimal_k = find_optimal_k(data, max_k=max_clusters)
    model = get_clustering_model(method, optimal_k)

    logger.info(f"Fitting {method} model with k={optimal_k}")
    labels = model.fit_predict(data)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Clustering model saved to {MODEL_PATH}")

    data_with_labels = data.copy()
    data_with_labels["Cluster"] = labels

    logger.info("Clustering complete. Cluster counts:")
    logger.info(data_with_labels["Cluster"].value_counts().to_dict())

    return data_with_labels


def load_saved_model():
    if MODEL_PATH.exists():
        logger.info(f"Loading clustering model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        logger.warning("No saved clustering model found.")
        return None
