from src.clustering import cluster_rfm_data  # type: ignore
from src.data_loader import load_rfm_data  # type: ignore
from src.preprocessing import preprocess_rfm  # type: ignore
from src.utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


def run_pipeline() -> None:

    logger.info("===== Starting Customer Segmentation Pipeline =====")

    raw_df = load_rfm_data()
    rfm_preprocessed = preprocess_rfm(raw_df)
    rfm_clustered = cluster_rfm_data(rfm_preprocessed)

    customer_id_col = "CustomerCode"
    output_df = raw_df[[customer_id_col]].copy()
    output_df["Cluster"] = rfm_clustered["Cluster"]

    logger.info("===== Pipeline execution complete =====")
    logger.info(f"Final output shape: {output_df.shape}")

    output_df.to_csv("data/customer_clusters.csv", index=False)
    logger.info("Clustered data saved to data/customer_clusters.csv")
