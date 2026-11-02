import pandas as pd
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)



class Cleaning:
    def __init__(self):
        logger.info("Data Understanding Pipeline Initialized")

    def inspect_dataframe(self , df:pd.DataFrame) -> pd.DataFrame:
        df=df.copy()

        logger.info("Shape: %s" , df.shape)
        logger.info("Columns: %s" , df.columns , list(df.columns))

        logger.info("First Five Rows: %s" , df.head())

        logger.info("Information: " )
        df.info()

        return df


    def check_missing_value(self,df:pd.DataFrame) -> pd.DataFrame:
        df=df.copy()


        missing_count=df.isnull().sum()


        missing_percentage=(
            df.isnull().sum() / len(df)
        ) * 100

        missing_report=pd.DataFrame({
            "Missing_Count" : missing_count,
            "Missing_Percentage" : missing_percentage
        })


        logger.info("Missing Report : %s" , missing_report)

        return df

    def identify_column_types(self,df:pd.DataFrame)->pd.DataFrame:
        df=df.copy()

        numeric_cols=df.select_dtypes(
            include=np.number 
        ).columns.to_list()

        categorical_cols=df.select_dtypes(
            include=["object","category","strings"]
        ).columns.to_list()

        datetime_cols = df.select_dtypes(
            include=["datetime"]
        ).columns.to_list()


        logger.info("Numerical: %s" , numeric_cols)
        logger.info("Categorical: %s" , categorical_cols)
        logger.info("Datetime: %s" , datetime_cols)


        return df

    def generate_summary(self , df:pd.DataFrame)->pd.DataFrame:
        df = df.copy()

        summary = df.describe(include=all)

        logger.info("Stats Summary: %s" , summary)

        return df 
    
    def fit_transform(self,df:pd.DataFrame)->pd.DataFrame:

        df_profiled = (
            df.pipe(self.inspect_dataframe).
            pipe(self.check_missing_value).
            pipe(self.identify_column_types).
            pipe(self.generate_summary)
        )

        return df_profiled
    
if __name__== "__main__":

    raw_data = {
        'Customer_ID': [101, 102, 103, 104, 104, 106, 107],
        'Age': [25, 30, np.nan, 45, 45, 150, 28],
        'Gender': [' Male ', 'Female', 'Male', 'Female',
                'Female', 'Male', 'female'],
        'Salary': [30000, 45000, 50000, np.nan, np.nan, 90000, 35000],
        'City': ['Delhi', 'Mumbai', 'Delhi', 'Pune',
                'Pune', None, 'Mumbai'],
        'Purchased': [1, 0, 1, 0, 0, 1, 1]
    }

    df_raw = pd.DataFrame(raw_data)

    print("RAW DATASET")
    print(df_raw)

    pipeline = Cleaning()

    df_profiled = pipeline.fit_transform(df_raw)

    print(df_profiled)
    