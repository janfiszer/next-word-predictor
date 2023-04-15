# word2vec parameters
VECTOR_SIZE = 10
WINDOW = 7
MIN_COUNT = 2
# unusual parameter
# determine how many words will be used to predict the next one
PREVIOUS_WORDS_CONSIDERED  = 3
END_TOKEN = '[END]'

# training parameters
EPOCHS = 64
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LABEL_SMOOTHING_RATIO = 0.1