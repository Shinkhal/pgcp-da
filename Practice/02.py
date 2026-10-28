from typing import Any

import pandas as pd
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    datefmt=r'%Y-%m-%d %H:%M:%S',
    format= '%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

filepath = './customer_subscription_sales_messy.csv'

def load_data(file: str)-> Any | None:
    try:
        df = pd.read_csv(file)
        logger.info(f"Loaded {file}")
        return df
    except FileNotFoundError:
        logger.error(f"File {file} not found")
        return None

class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def inspect_data(self):
        print(self.df.describe())
        print(self.df.dtypes)
        print(self.df.isnull().sum())

    def display_df(self):
        print(self.df)





if __name__ == "__main__":
    df = load_data(filepath)

    data_clean = DataCleaning(df)
    data_clean.inspect_data()
    data_clean.display_df()

