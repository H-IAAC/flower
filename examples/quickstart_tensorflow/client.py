import argparse
import os

from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HeadModel:
    def __init__(self, features, ACT_LABELS) -> None:
        self.n_features = len(features)
        self.n_labels = len(ACT_LABELS)
        self.base_model = self.build_base_model()

    def get_base_model(self) -> Sequential:
        return self.base_model

    def build_base_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(32, input_dim=self.n_features, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.n_labels, activation='softmax'))  # Softmax for multi-class classification
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


# Define Flower client
class ModelClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    features = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z', 'rotationRate.x', 'rotationRate.y',
                'rotationRate.z', 'attitude.roll', 'attitude.pitch', 'attitude.yaw']

    ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]

    model_obj = HeadModel(features, ACT_LABELS)
    model = model_obj.get_base_model()

    # Load local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = ModelClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    x_train = genfromtxt('./X_train.csv', delimiter=',')
    # filtrando a primeira coluna e a primeira linha (labels das colunas)
    x_train = x_train[1:,1:]

    x_test = genfromtxt('./X_test.csv', delimiter=',')
    # filtrando a primeira coluna e a primeira linha (labels das colunas)
    x_test = x_test[1:,1:]

    y_train = genfromtxt('./y_train.csv', delimiter=',')
    # filtrando a primeira coluna e a primeira linha (labels das colunas)
    y_train = y_train[1:,1:]

    y_test = genfromtxt('./y_test.csv', delimiter=',')
    # filtrando a primeira coluna e a primeira linha (labels das colunas)
    y_test = y_test[1:,1:]

    return (x_train, y_train,), (x_test, y_test,)

    """
    #Load 1/10th of the training and test data to simulate a partition.
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )
    """


if __name__ == "__main__":
    main()
