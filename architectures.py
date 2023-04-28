import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM

import config


def build_fnn(vocabulary_size: int):
    """
    Return classical, dense feedforward NN
    """
    model = Sequential()

    model.add(Input(config.INPUT_SIZE))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(vocabulary_size, activation="sigmoid"))

    model.build()

    return model


def build_rnn(vocabulary_size):
    """
    Returns RNN with a LSTM module
    """
    model = Sequential()

    model.add(LSTM(128, input_shape=(config.PREVIOUS_WORDS_CONSIDERED, config.VECTOR_SIZE)))
    model.add(Dense(vocabulary_size, activation="sigmoid"))

    model.build()

    return model
