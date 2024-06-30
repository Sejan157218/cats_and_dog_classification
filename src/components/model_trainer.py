import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()
    
    def initiate_model_trainer(self, training_set, test_set):
        print(training_set,test_set)
        try:
            logging.info("split training and test input data")

            cnn_base = tf.keras.applications.VGG16(
                        include_top=True,
                        weights="imagenet",
                        input_shape=(224, 224, 3),
                        pooling=None,
                        classes=1000,
                        classifier_activation="softmax",
                    )

            cnn = tf.keras.models.Sequential()
            cnn.add(cnn_base)
            # cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 3]))
            
            # cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

            # cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

            # cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

            cnn.add(tf.keras.layers.Flatten())

            cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

            cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

            cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

            cnn_base.trainable = False
            cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

            model = cnn.fit(x = training_set, validation_data = test_set, epochs = 1)
            logging.info(f"Best found mode on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )

            # predicted = best_model.predict(X_test)

            # r2_square = r2_score(y_test, predicted)

            # return r2_square

        except Exception as E:
            raise CustomException(E, sys)