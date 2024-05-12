import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from data_load import DataIngestionConfig, DataIngestion
from data_preprocess import DataTransformerConfig, DataTransformation
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import randint

# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


class ModelConfig:
    model_save_path: str = os.path.join("artifacts", "model.pkl")


class Model:
    def __init__(self, X, y,test_X,test_y) -> None:
        self.model_path_obj = ModelConfig()
        self.model_path = self.model_path_obj.model_save_path
        self.X = X.copy()
        self.y = y.copy()
        self.test_X = test_X.copy()
        self.test_y = test_y.copy()

    def define_model(self):
        self.model_map = {
            "Logestic regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision tree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "AdaboostClassifier": AdaBoostClassifier(),
            "SVC": SVC(),
            "XGBClassifier": XGBClassifier()
        }

    def model_train(self):
        for key, model in self.model_map.items():
            print(f"fitting {key,model} model")
            model.fit(self.X, self.y)

    def model_predict(self):
        self.pred = []
        self.test_pred = []
        for model in self.model_map.values():
            self.pred.append(model.predict(self.X))
            self.test_pred.append(model.predict(self.test_X))

    def model_eval(self):
        i = 0
        for key in self.model_map.keys():
            print(f"{key} model has {accuracy_score(self.y,self.pred[i])} accuracy")
            print(f"{key} model has {accuracy_score(self.test_y,self.test_pred[i])} accuracy")
            print(f"{key} model confusion matrix\n: {confusion_matrix(self.test_y,self.test_pred[i])}")
            print(f"{key} model has {precision_score(self.test_y,self.test_pred[i])} precision_score")
            print(f"{key} model has {recall_score(self.test_y,self.test_pred[i])} recall_score")
            print(f"{key} classification report of model\n: {classification_report(self.test_y,self.test_pred[i])} accuracy")
            print(f"{key} model has {f1_score(self.test_y,self.test_pred[i])} f1_score")
            print(f"{key} model has {roc_auc_score(self.test_y,self.test_pred[i])} roc_auc_score")
            i += 1


    def hyperparameter_tuning(self):

        param_grid = {
    "Logestic regression": {"C": [0.1, 1, 10, 100]},
    "KNN": {"n_neighbors": list(range(1, 30))},
    "Decision tree": {"max_depth": list(range(1, 10))},
    "RandomForest": {"n_estimators": list(range(10, 100)), "max_depth": list(range(1, 10))},
    "AdaboostClassifier": {"n_estimators": list(range(50, 100)), "learning_rate": [0.01, 0.1, 1]},
    "SVC": {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]},
    "XGBClassifier": {"n_estimators": list(range(50, 100)), "learning_rate": [0.01, 0.1, 1]}
        }

    # Perform randomized search for each model
        for model_name, model in self.model_map.items():
            print(f"Performing randomized search for {model_name}...")
            random_search = GridSearchCV(model, param_grid[model_name], cv=5)
            random_search.fit(self.X, self.y)
            self.model_map[model_name] = random_search
            print(f"Best parameters for {model_name}: {random_search.best_params_}")
        logging.info('Evaluation after hyperparameter tuning started')
        self.model_eval()
    def model_trainer(self):
        try:
            logging.info('Model training started')
            self.define_model()
            logging.info('Model defined')
            logging.info('Model training started')
            self.model_train()
            logging.info('Model prediction started')
            self.model_predict()
            logging.info('Model evaluation started')
            self.model_eval()
            logging.info('Hyperparameter tuning started')
            self.hyperparameter_tuning()
            logging.info('Hyperparameter tuning completed')
        except Exception as e:
            logging.info(f'Error occured {e}')
            raise CustomException(e,sys)


# if __name__ == "__main__":
#     try:
#         path_obj: DataIngestionConfig = DataIngestionConfig()
#         df_train: pd.DataFrame = pd.read_csv(path_obj.train_path)
#         obj: DataTransformation = DataTransformation(df_train)
#         X_train, y_trian = obj.preprocess_train_data()
#         df_test: pd.DataFrame = pd.read_csv(path_obj.test_path)
#         obj: DataTransformation = DataTransformation(df_test)
#         X_test, y_test = obj.preprocess_test_data()
#         model_obj = Model(X=X_train, y=y_trian,test_X=X_test,test_y=y_test)
#         model_obj.model_trainer()
#         logging.info("Completed")
#     except Exception as e:
#         logging.info(f"error occured {e}")
#         raise CustomException(e, sys)
