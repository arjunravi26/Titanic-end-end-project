import pandas as pd
import numpy as np
import os
import sys
from data_load import DataIngestionConfig
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PolynomialFeatures,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from typing import Tuple
from xgboost import XGBClassifier
from src.utils import save_object
from joblib import load


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

   
        self.df["Family_size"] = self.df["Parch"] + self.df["SibSp"] + 1
        family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small',
                    5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
        self.df['Family_Group'] = self.df['Family_size'].map(family_map)


    def feature_selection(self) -> None:
        # Select the important features for prediction
        self.df.drop(
            columns=["PassengerId", "Name", "SibSp",
                     "Parch", "Ticket", "Cabin",'Family_size','Age'],
            axis=1,
            inplace=True,
        )


    def create_pipeline(self) -> None:
        # for preprocessing raw data such as handle null value, encode and scale we create a pipeline here

        cat_cols_onehot: list = ["Sex"]
        cat_cols_ordinal: list = ["Age_Group",'Family_Group','Embarked']
        nums_cols: list = ["Fare"]

        # pipeline for 'Sex' feature
        cat_onehot_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot_encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        # pipeline for 'Age_Group' and 'Family Group'
        cat_ordinal_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(
                    categories=[self.age_labels, ['Alone', 'Small', 'Medium', 'Large'],['S','C','Q']])),
            ]
        )
        # pipeline for 'Fare'
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
        self.train_data_pipeline = self.pipeline.fit_transform(self.feature)



    def save_preproecssor(self):
        # save preprocessor pipeline object
        save_object(self.preprocessor_path, self.pipeline)
        
        
    def split_data(self) -> None:
        # split data into features and labels
        self.feature: pd.DataFrame = self.df.drop("Survived", axis=1)
        self.label: pd.Series = self.df["Survived"]    
        
        
    # for preprocessing training data
    def preprocess_train_data(self):
        self.feature_engineering()
        self.feature_selection()
        self.split_data()
        self.create_pipeline()
        self.save_preproecssor()
        return self.train_data_pipeline,self.label
    
    # for preprocessing test data
    def preprocess_test_data(self):
        self.feature_engineering()
        self.feature_selection()
        self.split_data()
        self.load_preprocessor = load(self.preprocessor_path)
        self.test_data = self.load_preprocessor.transform(self.feature)
        return self.test_data,self.label
    
    # for preprocessing new data data
    def preprocess_pred_data(self):
        self.feature_engineering()
        self.feature_selection()
        self.load_preprocessor = load(self.preprocessor_path)
        self.pred_data = self.load_preprocessor.transform(self.feature)
        return self.pred_data
