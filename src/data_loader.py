import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()
config = get_config()


def load_rfm_data() -> pd.DataFrame:

    try:
        logger.info("Starting to load RFM data from SQL")

        server = os.getenv("SQL_SERVER")
        database = os.getenv("SQL_DB")
        username = os.getenv("SQL_USER")
        password = os.getenv("SQL_PASS")

        driver = config["sql"]["driver"]
        table = config["sql"]["table"]

        conn_str = (
            f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
        )
        engine = create_engine(conn_str)

        logger.info(
            f"Connecting to SQL Server: {server}, Database: {database}, Table: {table}"
        )
        df = pd.read_sql(
            f"SELECT CustomerCode, RecencyEn AS Recency, Frequency, MonetaryValue, ReturnRate FROM {table}",
            engine,
        )

        logger.info(f"RFM data loaded successfully. Shape: {df.shape}")
        return df

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error while loading data: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading RFM data: {e}")
        raise
