# essentials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from collections import Counter
import pickle

## Dense NN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

# train test split 
from sklearn.model_selection import train_test_split

# my files
import config
from utils.DataGenerator import DataGenerator

# Parameters

small_data = True
seed = 42

# Loading the model
def load_data():
    # Loading the data

    if small_data:
        data_path = "data/reviews_cleaned_sample.csv"
        vocabulary_filename = "vocabulary_small.pkl"
    else:
        data_path = "data/reviews_cleaned.csv"
        vocabulary_filename = "vocabulary.pkl"

    reviews_df = pd.read_csv(data_path, converters={'tokenized': pd.eval})
    return reviews_df


def load_word2vec(word2vec_path):
    # Loading the word2vec model
    # Since I am using the whole dataset to create word2vec I can use the same one every time, regardless to the sample I take to train the NN

    with open(word2vec_path, "rb") as file:
        return pickle.load(file)


def create_dataset(documents, word_vectorizer):
    # Creating the dataset
    # Using custom class I am creating the dataset

    data_generator = DataGenerator(documents)
    data_generator.create_vocabulary(min_count=config.VOCAB_MIN_COUNT, extra_tokens=[config.END_TOKEN])

    vocabulary_size = len(data_generator.vocabulary)

    X_words, y_words = data_generator.create_dataset(previous_words_considered=config.PREVIOUS_WORDS_CONSIDERED)
    X = data_generator.vectorize(X_words, word_vectorizer=word_vectorizer, input_size=config.INPUT_SIZE)


    # one hotting the labels

    label_to_index = {word: index for index, word in enumerate(data_generator.vocabulary)}

    y = np.zeros((len(y_words), len(data_generator.vocabulary)))

    # instead of using for example to_categorical from tensorflow to make it more efficient 
    # I just use the created matrix of zeros and add 1 in appropriate places
    for index, label in enumerate(y_words):
        y[index, label_to_index[label]] += 1

    return X, y, vocabulary_size




# Dense NN model

def build_model(vocabulary_size):

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

    model.summary()

    return model


def training(model, X_train, y_train):
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING_RATIO)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])


    # patience and start_from_epochs depending on the epochs number
    # patience 10% of the total number of epochs
    # start_from_epochs 5% of the total number of epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=int(config.EPOCHS*0.10), 
                                                    start_from_epoch=(config.EPOCHS*0.05),
                                                    min_delta=0.001,
                                                    restore_best_weights=True,
                                                    verbose=1) 



    training = model.fit(X_train, y_train, validation_split=0.1, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, callbacks=[early_stopping])

    with open(os.path.join(config.NN_MODEL_DIR, "lr{config.LEARNING_RATE}-batch{config.BATCH_SIZE}training_logs.pkl"), "wb") as file:
        return pickle.dump(training, file)


def save(model, X_test, y_test):
    _, acc = model.evaluate(X_test, y_test)

    acc = int(acc * 100)
    model_filename = f"dense-lr{config.LEARNING_RATE}-batch{config.BATCH_SIZE}-acc{acc}.h5"

    model.save(os.path.join(config.NN_MODEL_DIR, model_filename))






def print_gap():
    print("---------------------------------------------------------------------------")
    print()
    print()


def main(model_dir):
    print("LOADING")
    try:
        os.makedirs(model_dir)
    except(FileExistsError):
        print(f"WARNING: You will may overwrite your models, because directory \"{model_dir}\" already exists.")

    reviews_df = load_data()

    word2vec = load_word2vec("models/word2vec-general.pkl")

    print_gap()

    print("CREATING DATASET")
    X, y, vocabulary_size = create_dataset(reviews_df.tokenized, word2vec)
    
    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=seed)

    print_gap()

    print("TRAINING MODEL")
    model = build_model(vocabulary_size)
    training(model, X_train, y_train)

    save(model, X_test, y_test)




if __name__ == "__main__":
    main(config.NN_MODEL_DIR)