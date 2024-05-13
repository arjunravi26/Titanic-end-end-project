from src.components.data_preprocess import DataTransformation
from src.components.model_trainer import ModelConfig
from joblib import load
import pandas as pd


class PredPipeline:
    def __init__(self) -> None:
        pass
    def pred_pipeline(self,data:pd.DataFrame):
        preprocess_obj = DataTransformation(data)
        data_processed = preprocess_obj.preprocess_pred_data()
        model_path = ModelConfig()
        model = load(model_path.model_save_path)
        self.pred = model.predict(data_processed)
        return self.pred
        
 
        
