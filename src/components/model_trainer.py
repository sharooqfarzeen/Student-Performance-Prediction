import os
import sys
from dataclasses import dataclass
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K - Neighbours Classifier": KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose = False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            # Construct the path to the hyperparameters.yaml file
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/hyperparameters.yaml'))
    
            with open(config_path, 'r') as file:
                logging.info("Loading hyperparameter file")
                param_grids = yaml.safe_load(file)
                logging.info(f'Hyperparameters {param_grids}')
            logging.info("Hyperparameters loaded")

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, param_grids)

            # Finding the model with the highest R² score
            best_model_name = max(model_report, key = model_report.get)

            # highest R² score
            best_score = model_report[best_model_name]
            # actual model function with highest score
            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No good models found")
            
            logging.info(f'The model with the highest R² score is: {best_model_name} with an R² score of {best_score}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # y_pred = best_model.predict(X_test)
            # r2_score = r2_score(y_test, y_pred)

            return best_model_name, best_score
        
        except Exception as e:
            raise CustomException(e, sys)