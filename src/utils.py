import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image


def load_and_preprocess_image(image):
    # image = Image.open(image)
    # image  = np.array(image)
    # resized_image = tf.image.resize(image, [224, 224])
    # # img_array = image.img_to_array(resized_image)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Rescale the image as per the training data preprocessing
    # return img_array

    image = Image.open(image)
    image  = np.array(image)
    resized_image = tf.image.resize(image, [224, 224])  # Example size, adjust as needed
    normalized_image = resized_image / 255.0  # Normalize pixel values
    input_image = tf.expand_dims(normalized_image, 0)  # Add batch dimension
    return input_image

def predict_image(model, img_array):
    prediction = model.predict(img_array)
    return prediction






