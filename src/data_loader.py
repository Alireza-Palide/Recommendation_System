import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def load_rfm_data():
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DB")
    username = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASS")

    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)

    query = "SELECT CustomerCode, Name, EntryDateEn, ReturnRate, RecencyEn, Frequency, MonetaryValue FROM RFM.CustomerRFM"
    df = pd.read_sql(query, engine)
    return df
