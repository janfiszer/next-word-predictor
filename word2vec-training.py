import config
import pandas as pd
import os
import pickle

from gensim.models import Word2Vec

data_path = "data/reviews_cleaned.csv"
reviews_df = pd.read_csv(data_path, converters={'tokenized': pd.eval})

word2vec = Word2Vec(reviews_df.tokenized,
                    min_count=config.WORD2VEC_MIN_COUNT,
                    window=config.WINDOW,
                    vector_size=config.VECTOR_SIZE)


# saving word2vec model
word2vec_filepath = f"{config.WORD2VEC_DIR}/whole-dataset-win7-vec200-min20.pkl"

with open(word2vec_filepath, 'wb') as file:
    pickle.dump(word2vec, file)

print("Word2vec model saved to {}".format(word2vec_filepath))
