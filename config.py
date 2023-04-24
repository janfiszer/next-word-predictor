# randomization
SEED = 42

# vocabulary
VOCAB_MIN_COUNT = 10

# word2vec parameters
VECTOR_SIZE = 200
WINDOW = 7
WORD2VEC_MIN_COUNT = 10

# unusual parameter
# determine how many words will be used to predict the next one
PREVIOUS_WORDS_CONSIDERED = 3
END_TOKEN = '[END]'

# for the NN
INPUT_SIZE = PREVIOUS_WORDS_CONSIDERED * VECTOR_SIZE

# training parameters
EPOCHS = 64
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LABEL_SMOOTHING_RATIO = 0.2

# directories
BASE_DIR = "models"
NN_MODEL_DIR = f"{BASE_DIR}/dense/min{VOCAB_MIN_COUNT}-pwc{PREVIOUS_WORDS_CONSIDERED}-lr{LEARNING_RATE}-batch{BATCH_SIZE}/"
WORD2VEC_DIR = f"{BASE_DIR}/word2vecs/"
