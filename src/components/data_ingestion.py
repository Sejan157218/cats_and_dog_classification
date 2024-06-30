import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformaion

from src.components.model_trainer import ModeltrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "training_set")
    test_data_path: str=os.path.join("artifacts", "test_set")



class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or components")
        try:
            train_source_folder = '/media/dev/100E83900E836D92/work/ML_PROJECT/dataset/training_set'
            train_destination_folder = self.ingestion_config.train_data_path

            test_source_folder = '/media/dev/100E83900E836D92/work/ML_PROJECT/dataset/test_set'
            test_destination_folder = self.ingestion_config.test_data_path

            logging.info("read the dataset as dataframe")
            # shutil.copytree(train_source_folder, train_destination_folder, dirs_exist_ok=True)
            # shutil.copytree(test_source_folder, test_destination_folder, dirs_exist_ok=True)

            logging.info("train_test_split initiated")

            logging.info("ingestion of data is completed")

            return{
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }
        except Exception as E:

            raise CustomException(E,sys)



if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()   
    data_transformation = DataTransformaion()
    train_arr, test_arr, _ = data_transformation.get_data_transformer_object()

    print(test_arr)

    modelTrainer = ModelTrainer()
    model_score = modelTrainer.initiate_model_trainer(train_arr, test_arr)

    # print("model_score", model_score)
