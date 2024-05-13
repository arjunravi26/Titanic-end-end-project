import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from src.utils import save_object



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
            "GradientBoostingClassifier":GradientBoostingClassifier()
        }

    def train(self):
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
    "XGBClassifier": {"n_estimators": list(range(50, 100)), "learning_rate": [0.01, 0.1, 1]},
    "GradientBoostingClassifier": {'n_estimators': [100, 200, 300],'learning_rate': [0.01, 0.1, 1],'max_depth': [3, 4, 5]}
}


    # Perform randomized search for each model
        for model_name, model in self.model_map.items():
            print(f"Performing Grid search for {model_name}...")
            grid_search = GridSearchCV(model, param_grid[model_name], cv=5)
            grid_search.fit(self.X, self.y)
            self.model_map[model_name] = grid_search
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logging.info('Evaluation after hyperparameter tuning started')
        self.model_eval()
    def save_model(self):
        name = input('Enter name of the model to save')
        logging.inof(f'{name},{self.model_map[name]} model saved in path {self.model_path}')
        save_object(self.model_path,self.model_map[name])
    def model_trainer(self):
        try:
            self.define_model()
            self.train()
            self.model_predict()
            self.model_eval()
            self.hyperparameter_tuning()
            self.save_model()
        except Exception as e:
            logging.info(f'Error occured {e}')
            raise CustomException(e,sys)



