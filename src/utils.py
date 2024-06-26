import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as obj:
            return pickle.load(obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        logging.info("Model evaluation initiated")
        #take each model name
        for model_name in models:
          model = models[model_name]
          
          # Hyperparameter Tuning
          best_model = GridSearchCV(model, params[model_name], cv=5, n_jobs=-1, scoring='r2')
          best_model.fit(X_train, y_train)

          y_train_pred = best_model.predict(X_train)

          y_test_pred = best_model.predict(X_test)

          train_model_score = r2_score(y_train, y_train_pred) 

          test_model_score = r2_score(y_test, y_test_pred)

          report[model_name] = test_model_score
        logging.info("Model evaluation complete")

        return report
     
    except Exception as e:
        raise CustomException(e, sys)