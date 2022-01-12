import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import flwr as fl
warnings.filterwarnings('ignore')
from data import Data
from harclient import HarClient

class HarModel(tf.keras.Sequential):
    def __init__(self) -> None:
        super().__init__()

        self.data = Data()
        self.init_model()
        #self.harclient = HarClient()


    def init_model(self):

        self.add(Dense(units=64,kernel_initializer='normal',activation='sigmoid',input_dim=self.data.x_train.shape[1]))
        self.add(Dropout(0.2))
        self.add(Dense(units=6,kernel_initializer='normal',activation='softmax'))
        self.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    def build_model(self, hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 25)):
            model.add(layers.Dense(units = hp.Int('units' + str(i), min_value=32, max_value=512, step=32),
                                kernel_initializer= hp.Choice('initializer', ['uniform', 'normal']),
                                activation= hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])))
        model.add(layers.Dense(6, kernel_initializer= hp.Choice('initializer', ['uniform', 'normal']), activation='softmax'))
        model.add(
                Dropout(0.2))
        model.compile(
            optimizer = 'adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model

    def tune_model(self):
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials= 5,
            executions_per_trial=3,
            directory='project', project_name = 'Human_activity_recognition')

        tuner.search_space_summary()
        self.model=tuner.get_best_models(num_models=1)[0]


# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")
model = HarModel()
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('harmodel.tflite', 'wb') as f:
  f.write(tflite_model)