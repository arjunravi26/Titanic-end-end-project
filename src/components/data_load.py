import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    data_path: str = os.path.join("artifacts", "data.csv")
    source_path: str = os.path.join("research", "Data", "train.csv")
    random_state: int = 44
    test_size: float = 0.2


class DataIngestion:
    def __init__(self):
        self.ingestion = DataIngestionConfig()

    def read_data(self) -> pd.DataFrame:
        """Reads data from the source file"""
        logging.info("Data reading started")
        os.makedirs(os.path.dirname(self.ingestion.train_path), exist_ok=True)
        df = pd.read_csv(self.ingestion.source_path)
        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split the data into traing and testing data"""
        logging.info("Splitting data started")
        train_df, test_df = train_test_split(
            df,
            random_state=self.ingestion.random_state,
            test_size=self.ingestion.test_size,
        )
        return (train_df, test_df)

    def write_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Writes the trianing and testing data to files"""
        logging.info("Writing data started")
        train_df.to_csv(self.ingestion.train_path, header=True, index=False)
        logging.info(f"Wrote {train_df.shape[0]} rows of training data")
        test_df.to_csv(self.ingestion.test_path, header=True, index=False)
        logging.info(f"Wrote {test_df.shape[0]} rows of testing data")

        return (self.ingestion.train_path, self.ingestion.test_path)

    def data_ingestion(self) -> tuple:
        """Perform data ingestion"""
        logging.info("Data ingestion started")
        try:
            data = self.read_data()
            train_df, test_df = self.split_data(data)
            path = self.write_data(train_df, test_df)
            logging.info("Data ingestion finished")
            return path

        except Exception as e:
            logging.info(f"Error during data ingestion {e}")
            raise CustomException(e, sys)
