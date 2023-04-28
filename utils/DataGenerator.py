import numpy as np
from typing import Iterable, Tuple
from collections import Counter

from utils.exceptions.NotInVocabularyException import NotInVocabularyException


class DataGenerator():
    def __init__(self, tokenized_documents: Iterable, vocabulary=None):
        """
        Class that allows to create the desired dataset from set of documents.
        """
        self.documents = tokenized_documents

        if vocabulary is not None:
            self.vocabulary = vocabulary

    def create_vocabulary(self, min_count=1, extra_tokens=[]) -> list:
        """ 
        Return and creates an attribute vocabulary, which is a list of unique words that occur in the documents.
        Params:
        min_count - minimum occurrence number to be included in the vocabulary
        extra_tokens - extra tokens that we want to add to the vocabulary e.g. EOF token
        """
        all_words = []

        for tokenized_reviews in self.documents:
            all_words.extend(tokenized_reviews)

        vocabulary = [word for word, quantity in Counter(all_words).items() if quantity >= min_count]

        vocabulary.extend(extra_tokens)

        self.vocabulary = vocabulary

    def create_dataset(self, previous_words_considered=3):
        """ 
        Returns tuple with two lists. First words (previous_words_considered determine how many) and in the second the following word
        """
        X_words = []
        y_words = []

        for tokenized_review in self.documents:
            # iterating over indices for the first which has enough words before (PREVIOUS_WORDS_CONSIDERED)
            # until the last one
            for index in range(previous_words_considered, len(tokenized_review)): 
                x = tokenized_review[index-previous_words_considered: index]
                y = tokenized_review[index]

                # only from token in vocabulary
                if y in self.vocabulary:
                    if self.belong_to_vocabulary(x) is None:
                        X_words.append(tokenized_review[index-previous_words_considered: index])
                        y_words.append(tokenized_review[index])

        return X_words, y_words

    def vectorize(self, X_words, word_vectorizer, input_size: Tuple):
        """ 
        Converts a 2D list of tokens into a 2D array of size where each row is a concatenation of
        given tokens, described by input_size = previous_words_considered * vector_size (see config.py).
        """
        try:
            input_shape = (len(X_words), )
            input_shape += input_size
        except TypeError:
            print(F"Expected a tuple as input_size, got {type(input_size)}")
            return

        X = np.zeros(input_shape)

        if len(input_shape) == 3:
            concatenate_axis = 0
        elif len(input_shape) == 2:
            concatenate_axis = 1
        # TODO: Exception?
        else:
            return None

        for index, words in enumerate(X_words):
            # initialization the numpy array with the embedding of the first word
            # idea!! first iterate over the word and create their embeddings
            # then np.concatenate them with appropiate axis
            embedding_list = [[word_vectorizer.wv.get_vector(word)] for word in words]
            embedded_np_array = np.concatenate(embedding_list, axis=concatenate_axis)


            # numpy_x = word_vectorizer.wv.get_vector(words[0])
            # for i in range(1, len(words)):
            #     # the mistake is here it hase to be with the vector dim , still not working should be debugged
            #     try:
            #         embedded_word = word_vectorizer.wv.get_vector(words[i])
            #         numpy_x = np.concatenate((numpy_x, embedded_word), axis=concatenate_axis)
            #     except KeyError:
            #         raise NotInVocabularyException

            X[index, ...] = embedded_np_array

        return X

    # TODO: rename
    def vectorize_lstm(self, X_words, word_vectorizer, vector_size, previous_word_considered):

        X = np.zeros((len(X_words), vector_size, previous_word_considered))

        for index, x in enumerate(X_words):
            numpy_x = np.array([])
            for word in x:
                try:
                    numpy_x = np.append(numpy_x, word_vectorizer.wv.get_vector(word))

                except KeyError:
                    raise NotInVocabularyException

            X[index, ...] = numpy_x

        return X

    def belong_to_vocabulary(self, tokens: list):
        """ 
        Returns the first not belonging token in the vocabulary
        """
        for token in tokens:
            if not token in self.vocabulary:
                return token
            
        return None