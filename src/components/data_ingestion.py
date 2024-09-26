import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import shutil
# from src.components.data_transformation import DataTransformationConfig
# from src.components.data_transformation import DataTransformaion

# from src.components.model_trainer import ModeltrainerConfig
# from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train")
    test_data_path: str=os.path.join("artifacts", "validation")



class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()


    def separate_images(self,source_dir, cats_dir, dogs_dir):
        # Ensure destination directories exist
        if not os.path.exists(cats_dir):
            os.makedirs(cats_dir)
        if not os.path.exists(dogs_dir):
            os.makedirs(dogs_dir)
        
        # Iterate through files in the source directory
        for filename in os.listdir(source_dir):
            # Check if the file is an image (optional, add more extensions if needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Determine the destination directory based on filename
                if 'cat' in filename.lower():
                    shutil.move(os.path.join(source_dir, filename), os.path.join(cats_dir, filename))
                elif 'dog' in filename.lower():
                    shutil.move(os.path.join(source_dir, filename), os.path.join(dogs_dir, filename))
                else:
                    print(f"Skipping {filename}: not recognized as a cat or dog image")

    def separate_velidation_images(self, source_dir, move_dir):
        # Ensure destination directories exist
        if not os.path.exists(move_dir):
            os.makedirs(move_dir)

        # Iterate through files in the source directory
        for filename in os.listdir(source_dir)[10000:]:
            # Check if the file is an image (optional, add more extensions if needed)
            shutil.move(os.path.join(source_dir, filename), os.path.join(move_dir, filename))
        print("Done")


    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or components")
        try:
            source_file_dir = "artifacts/train"
            cats_dir = "artifacts/train/cats"
            dogs_dir = "artifacts/train/dogs"
            data_separate = self.separate_images(source_file_dir, cats_dir, dogs_dir)
            source_dir = "artifacts/train/cats"
            move_dir = "artifacts/validation/cats"
            self.separate_velidation_images(source_dir,move_dir)

            source_dir_dogs = "artifacts/train/dogs"
            move_dir_dogs = "artifacts/validation/dogs"
            self.separate_velidation_images(source_dir_dogs,move_dir_dogs)

            logging.info("ingestion of data is completed")

            return{
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }
        except Exception as E:

            raise CustomException(E,sys)



# if __name__=="__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()   
#     # data_transformation = DataTransformaion()
#     # train_arr, test_arr, _ = data_transformation.get_data_transformer_object()

#     # print(test_arr)

#     # modelTrainer = ModelTrainer()
#     # model_score = modelTrainer.initiate_model_trainer()

#     # print("model_score", model_score)
