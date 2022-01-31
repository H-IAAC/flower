import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras_tuner.engine.hyperparameters import HyperParameters
import os
import warnings
from tensorflow import keras
from tensorflow.keras import layers

# HAR model
from harmodel import HarModel

import flwr as fl
import tensorflow as tf
warnings.filterwarnings('ignore')

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    # Load and compile model
    model = HarModel()

    # Define Flower client
    class HarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(model.data.x_train, model.data.y_train, epochs=1, batch_size=64, steps_per_epoch=3)
            return model.get_weights(), len(model.data.x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(model.data.x_test, model.data.y_test)
            return loss, len(model.data.x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=HarClient())
