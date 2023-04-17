import numpy as np
from collections import Counter

class DataGenerator():
    def __init__(self, tokenized_documents):
        self.documents = tokenized_documents


    def create_vocabulary(self, min_count=1, extra_tokens=[]) -> list:
        """ Return and creates an attribute vocabulary, which is a list of unique words that occur in the documents.
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
        """ Returns tuple with two lists. First words (previous_words_considered determine how many) and in the second the following word
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
                    if all(xx in self.vocabulary for xx in x):
                        X_words.append(tokenized_review[index-previous_words_considered: index])
                        y_words.append(tokenized_review[index])

        return (X_words, y_words)
    
    def vectorize(self, X_words, y_words, word_vectorizer, input_size):
        label_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        
        X = np.zeros((len(X_words), input_size))

        for index, x in enumerate(X_words):
            numpy_x = np.array([])
            for word in x:
                numpy_x = np.append(numpy_x, word_vectorizer.wv.get_vector(word))

            X[index, ...]  = numpy_x

        y = np.zeros((len(y_words), len(self.vocabulary)))

        for index, label in enumerate(y_words):
            y[index, label_to_index[label]] += 1

        return X, y
            
