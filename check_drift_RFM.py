import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.config_loader import get_config
from src.data_loader import load_rfm_data
from src.pipeline import run_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

DRIFT_FILE = Path("models/feature_stats.json")
DRIFT_THRESHOLD = config.get("drift", {}).get("threshold", 0.2)


def extract_raw_features(df_raw: pd.DataFrame) -> pd.DataFrame:

    cols = config["rfm_columns"]
    ref_date = datetime.strptime(config["recency_reference_date"], "%Y-%m-%d")
    entry_dates = pd.to_datetime(df_raw[cols["recency"]], errors="coerce")
    recency_days = (ref_date - entry_dates).dt.days

    features = pd.DataFrame(
        {
            "Recency": recency_days,
            "Frequency": df_raw[cols["frequency"]],
            "Monetary": df_raw[cols["monetary"]],
            "ReturnRate": df_raw[cols["return_rate"]],
        }
    )

    return features.dropna()


def compute_feature_means(df: pd.DataFrame) -> dict:
    return df.mean(numeric_only=True).to_dict()


def has_drift(
    previous: dict, current: dict, threshold: float = DRIFT_THRESHOLD
) -> bool:
    for feature in current:
        if feature in previous:
            old = previous[feature]
            new = current[feature]
            if old == 0:
                continue
            change = abs(new - old) / abs(old)
            if change > threshold:
                logger.info(f"Drift detected in '{feature}': change = {change:.2%}")
                return True
    return False


def load_previous_stats() -> dict:
    if DRIFT_FILE.exists():
        with open(DRIFT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_current_stats(stats: dict):
    Path("models").mkdir(exist_ok=True)
    with open(DRIFT_FILE, "w") as f:
        json.dump(stats, f)


def main():
    logger.info("Starting drift check...")

    df_raw = load_rfm_data()
    df_features = extract_raw_features(df_raw)
    current_stats = compute_feature_means(df_features)
    previous_stats = load_previous_stats()

    if not previous_stats:
        logger.info(
            "No previous stats found. Saving current stats for future comparison."
        )
        save_current_stats(current_stats)
        return

    if has_drift(previous_stats, current_stats):
        logger.info("Drift confirmed. Retraining pipeline...")
        run_pipeline()
        save_current_stats(current_stats)
        logger.info("Model retrained and stats updated.")
    else:
        logger.info("No significant drift detected. No action taken.")


if __name__ == "__main__":
    main()
