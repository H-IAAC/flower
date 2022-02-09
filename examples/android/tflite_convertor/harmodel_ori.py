import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

warnings.filterwarnings('ignore')
from data import Data


class HarModel(tf.keras.Sequential):
    def __init__(self) -> None:
        super().__init__()

        self.data = Data()
        self.init_model()
        # self.harclient = HarClient()

    def init_model(self):
        self.add(
            Dense(units=64, kernel_initializer='normal', activation='sigmoid', input_dim=self.data.x_train.shape[1]))
        self.add(Dropout(0.2))
        self.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def build_model(self, hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 25)):
            model.add(layers.Dense(units=hp.Int('units' + str(i), min_value=32, max_value=512, step=32),
                                   kernel_initializer=hp.Choice('initializer', ['uniform', 'normal']),
                                   activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])))
        model.add(
            layers.Dense(6, kernel_initializer=hp.Choice('initializer', ['uniform', 'normal']), activation='softmax'))
        model.add(
            Dropout(0.2))
        model.compile(
            optimizer='adam',
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


"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32   

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""


head = tf.keras.Sequential(
    [
        tf.keras.Input(model.data.x_train.shape[1]),
        tf.keras.layers.Dense(units=64, kernel_initializer='normal', activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=6, kernel_initializer='normal', activation='softmax'),
    ]
)

#head.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
head.compile(optimizer='adam', loss='categorical_crossentropy')

"""Convert the model for TFLite.

Using 6 classes in HAR, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""

base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    6, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-3), train_batch_size=64
)

converter.convert_and_save("tflite_model")
