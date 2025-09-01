from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.config_loader import get_config  # type: ignore
from src.utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)
config = get_config()


def preprocess_rfm(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting RFM preprocessing")

    cols = config["rfm_columns"]

    entry_dates = pd.to_datetime(df[cols["recency"]], errors="coerce")
    reference_date = datetime.strptime(config["recency_reference_date"], "%Y-%m-%d")
    recency_days = (reference_date - entry_dates).dt.days

    features = pd.DataFrame(
        {
            "Recency": recency_days,
            "Frequency": df.loc[recency_days.index, cols["frequency"]],
            "Monetary": df.loc[recency_days.index, cols["monetary"]],
            "ReturnRate": df.loc[recency_days.index, cols["return_rate"]],
        }
    )

    if features.isnull().sum().sum() > 0:
        logger.warning("Missing values found in feature data. Dropping them.")
        features = features.dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info(
        "Feature preprocessing complete. Shape after scaling: %s", features_scaled.shape
    )

    return pd.DataFrame(features_scaled, columns=features.columns)
