import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class CustomData:
    def __init__(self, writing_score: int, reading_score: int, gender: str, race_ethnicity: str, 
                 parental_level_of_education: str, lunch: str, test_preparation_course: str):
        self.writing_score = writing_score
        self.reading_score = reading_score
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def get_dataframe(self):
        try:
            data = {'writing_score': [self.writing_score],
                    'reading_score': [self.reading_score],
                    'gender': [self.gender],
                    'race_ethnicity': [self.race_ethnicity],
                    'parental_level_of_education': [self.parental_level_of_education],
                    'lunch': self.lunch,
                    'test_preparation_course': [self.test_preparation_course]
                    }
            
            return pd.DataFrame(data)
        
        except Exception as e:
            raise CustomException(e, sys)
        
class Predict:
    def __init__(self):
        pass

    def predict(self, features):
        logging.info("Loading preprocessor and model objects")
        pre_processor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
        model = load_object(os.path.join('artifacts', 'model.pkl'))
        logging.info("Preprocessor and model loaded")

        logging.info("Transforming data")
        processed_data = pre_processor.transform(features)
        logging.info("Predicting")
        prediction = model.predict(processed_data)

        return prediction