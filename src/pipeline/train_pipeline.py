from src.components.data_load import DataIngestion,DataIngestionConfig
from src.components.data_preprocess import DataTransformation
from src.components.model_trainer import Model
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
class TrainPipeline:
    def __init__(self) -> None:
        pass
    def train_pipeline(self):
        self.data_obj = DataIngestion()
        self.data_path = DataIngestionConfig()
        train_data = pd.read_csv(self.data_path.train_path)
        preprocess_obj_train = DataTransformation(train_data)
        test_data = pd.read_csv(self.data_path.test_path)
        preprocess_obj_test = DataTransformation(test_data)
        X_train,y_train = preprocess_obj_train.preprocess_train_data()
        X_test,y_test = preprocess_obj_test.preprocess_test_data()
        model = Model(X=X_train,y=y_train,test_X=X_test,test_y=y_test)
        model.model_trainer()
        
if __name__ == "__main__":
    try:
        obj = TrainPipeline()
        obj.train_pipeline()
    except Exception as e:
        logging.info(f"error occured {e}")
        raise CustomException(e, sys)
        
        