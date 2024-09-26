import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_and_preprocess_image, predict_image
import tensorflow as tf


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,image):
        try:
            model =  tf.keras.models.load_model('artifacts/cats_dogs_classifier.h5')

            img_array = load_and_preprocess_image(image)
            prediction = predict_image(model, img_array)

            # Assuming your model outputs a single probability score
            result = ''
            if prediction[0] > 0.5:
                result = 'Dog'
            else:
                result = 'Cat'
            return result
        
        except Exception as e:
            raise CustomException(e,sys)
