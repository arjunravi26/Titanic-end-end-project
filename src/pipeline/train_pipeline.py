from src.components.data_load import DataIngestionConfig
from src.components.data_preprocess import DataIngestionConfig,DataTransformation
from src.components.model_trainer import Model

class TrainPipeline:
    def __init__(self) -> None:
        pass
    def train_pipeline(self):
        self.data_path = DataIngestionConfig()
        preprocess_obj = DataTransformation()
        X_train,y_train = preprocess_obj.preprocess_train_data()
        X_test,y_test = preprocess_obj.preprocess_test_data()
        model = Model(X=X_train,y=y_train,test_X=X_test,test_y=y_test)
        model.model_trainer()
        
        