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

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
            # Define directories
            train_dir = 'artifacts/train'
            validation_dir = 'artifacts/validation'

            # Data augmentation and rescaling for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Only rescaling for validation
            validation_datagen = ImageDataGenerator(rescale=1./255)

            # Create data generators
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=20,
                class_mode='binary'
            )

            validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(224, 224),
                batch_size=20,
                class_mode='binary'
            )

            return (
                train_generator,
                validation_generator
            )


        except Exception as E:
            raise CustomException(E, sys)



# if __name__=="__main__":
#     obj = DataTransformaion()
#     data= obj.get_data_transformer_object()   