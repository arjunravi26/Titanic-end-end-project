from src.components.data_load import DataIngestionConfig
from src.components.data_preprocess import DataIngestionConfig,DataTransformation
from src.components.model_trainer import ModelConfig
from joblib import load


class PredPipeline:
    def __init__(self) -> None:
        pass
    def pred_pipeline(self):
        self.data_path = DataIngestionConfig()
        preprocess_obj = DataTransformation()
        data = preprocess_obj.preprocess_pred_data()
        model_path = ModelConfig()
        model = load(model_path.model_save_path)
        self.pred = model.predict(data)
        
 
        
