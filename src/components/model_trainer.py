import os
import sys
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class ModeltrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "c_a_d_classifier.h5")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()
    
    def initiate_model_trainer(self, train_generator, validation_generator):
        print("model train start")
        try:
            # Check TensorFlow version and GPU availability
            print(tf.__version__)
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

                        
            model = Sequential()

            model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(224,224,3)))

            model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

            model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))

            model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

            model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))

            model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

            model.add(Flatten())

            model.add(Dense(256,activation='relu'))

            model.add(Dense(1,activation='sigmoid'))

            model.summary()

            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(
                train_generator,
                # steps_per_epoch=400,  # 2000 images = batch_size * steps
                epochs=15,
                validation_data=validation_generator,

                callbacks=[early_stopping]
                # validation_steps=125  # 1000 images = batch_size * steps
            )

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj = model
            # )
            model.save("artifacts/cats_dogs_classifier.h5")
        except Exception as E:
            raise CustomException(E, sys)

# if __name__=="__main__":
#     obj = ModelTrainer()
#     data= obj.initiate_model_trainer()  