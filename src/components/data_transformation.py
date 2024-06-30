import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts_file', 'preprocessor.pkl')
    train_generate_file = os.path.join('artifacts_generate_image', 'preprocessor.pkl')


class DataTransformaion:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformer_object(self):

        """

        This function is responsible for data transformation

        """
        try:
            # Set up the directory for saving augmented images
            save_dir = 'augmented_images'
            os.makedirs(save_dir, exist_ok=True)

            # Create the ImageDataGenerator with augmentation options
            train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            # save_to_dir=save_dir,  # Directory to save augmented images
            # save_prefix='aug',    # Prefix for saved images
            # save_format='jpeg'    # Format to save images
            )

            # Create a data generator
            train_generator = train_datagen.flow_from_directory(
                'artifacts/training_set', 
                target_size=(224, 224), 
                batch_size=32, 
                class_mode='binary',
                # save_to_dir=save_dir,
                # save_prefix='aug',
                )

            test_datagen = ImageDataGenerator(rescale = 1./255)
            test_generator = test_datagen.flow_from_directory('artifacts/test_set',
                                                        target_size = (224, 224),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

            # Generate a few batches of augmented images and save them
            # num_batches = 10  # Number of batches to generate and save
            # for i in range(num_batches):
            #     batch = next(train_generator)

            # logging.info(f"Saved preprocessing object . ")

            # save_object(
            #     file_path = self.data_transformation_config.preprocessor_obj_file_path,
            #     obj = preprocessing_obj
            # )

            return (
                train_generator,
                test_generator,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as E:
            raise CustomException(E, sys)



if __name__=="__main__":
    obj = DataTransformaion()
    data= obj.get_data_transformer_object()   