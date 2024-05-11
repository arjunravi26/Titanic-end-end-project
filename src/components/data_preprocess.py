import pandas as pd
import numpy as np
import os
import sys
from data_load import DataIngestionConfig
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from typing import Tuple
from src.utils import save_object

# this class have the path for storing preprocessor pipeline pickle file
@dataclass
class DataTransformerConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "pipeline.pkl")


class DataTransformation:
    """
    This class is responsible for data transformation
    """

    def __init__(self, data) -> None:
        # here proproecssor pipeline path and dataframeis stored
        logging.info('Data preproessing started')
        obj: DataTransformerConfig = DataTransformerConfig()
        self.preprocessor_path: str = obj.preprocessor_obj_path
        self.df: pd.DataFrame = data.copy()

    def feature_engineering(self) -> None:
        # new features are extracted from teh existing ones.
        # age group
        age_bins: list = [
            self.df["Age"].min(), 5, 13, 18, 30, 40, 56, self.df["Age"].max() + 1,
        ]
        self.age_labels: list = [
            "Baby", "Child", "Teenager", "Young Adult", "Adult", "Middle-Aged", "Senior",
        ]
        self.df["Age_Group"] = pd.cut(
            self.df["Age"], bins=age_bins, labels=self.age_labels, right=False
        )

        # child with parents
        self.df["ChildWithParents"] = np.where(
            (self.df["Age"] < 18) & (self.df["Parch"] > 0), 1, 0
        )

    def feature_selection(self) -> None:
        # Select the important features for prediction
        self.df.drop(
            columns=["PassengerId", "Name", "SibSp",
                     "Parch", "Ticket", "Cabin"],
            axis=1,
            inplace=True,
        )

    def create_pipeline(self) -> None:
        # for preprocessing raw data such as handle null value, encode and scale we create a pipeline here

        cat_cols_onehot: list = ["Sex"]
        cat_cols_ordinal: list = ["Age_Group", "Embarked"]
        nums_cols: list = ["Age", "Fare"]

        # pipeline for 'Sex' feature
        cat_onehot_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot_encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        # pipeline for 'Age_Group' and 'Embarked'
        cat_ordinal_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(
                    categories=[self.age_labels, ["S", "C", "Q"]])),
            ]
        )
        # pipeline for 'Age' and 'Fare'
        nums_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        # preprocessor store the pipelines and its cols name
        preprocessor: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("cat_onehot_pipeline", cat_onehot_pipeline, cat_cols_onehot),
                ("cat_ordinal_pipeline", cat_ordinal_pipeline, cat_cols_ordinal),
                ("num_pipeline", nums_pipeline, nums_cols),
            ]
        )
        # final pipeline is created using the ColumnTransfomer
        self.pipeline: Pipeline = Pipeline(steps=[("pipeline", preprocessor)])

    def split_data(self) -> None:
        # split data into features and labels
        self.feature: pd.DataFrame = self.df.drop("Survived", axis=1)
        self.label: pd.Series = self.df["Survived"]

    def save_preproecssor(self):
        # save preprocessor pipeline object
        save_object(self.preprocessor_path, self.pipeline)

    def preprocess_data(self) -> Tuple[np.ndarray, pd.Series]:
        # all above proprocessing functions are in order
        try:
            logging.info('Feature engineering started')
            self.feature_engineering()
            logging.info('Feature selection started')
            self.feature_selection()
            logging.info('Pipeline creation started')
            self.create_pipeline()
            logging.info('Data splitting started')
            self.save_preproecssor()
            logging.info(
                f'Preprocessor pipeline saved as {self.preprocessor_path}')
            self.split_data()
            logging.info('Data transformation started')
            self.processed_feature = self.pipeline.fit_transform(self.feature)
            logging.info('Data preprocessing finished')
            return self.processed_feature, self.label
        except Exception as e:
            logging.info(f'Error Occured {e,sys}')
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        path_obj: DataIngestionConfig = DataIngestionConfig()
        df: pd.DataFrame = pd.read_csv(path_obj.train_path)
        obj: DataTransformation = DataTransformation(df)
        X, y = obj.preprocess_data()
        logging.info('Completed')
    except Exception as e:
        logging.info(f'error occured {e}')
        raise CustomException(e, sys)
