import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            logging.info("Reading dataset")
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read dataset to dataframe")

            logging.info("Train test split inititated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            logging.info("Train test split completed")

            logging.info("Creating required directories")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            logging.info("Directories Created")
            
            logging.info("Writing raw data, training data and test data to ingestion pipeline")
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data ingestion complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    best_model, best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f'The model with the highest R² score is: {best_model} with an R² score of {best_score}')