# word2vec parameters
VECTOR_SIZE = 200
WINDOW = 7
MIN_COUNT = 20
# unusual parameter
# determine how many words will be used to predict the next one
PREVIOUS_WORDS_CONSIDERED  = 3
END_TOKEN = '[END]'

# for the NN
INPUT_SIZE = PREVIOUS_WORDS_CONSIDERED * VECTOR_SIZE

# training parameters
EPOCHS = 64
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LABEL_SMOOTHING_RATIO = 0.2

# directories
BASE_DIR = "models/word2vecs/"
NN_MODEL_DIR = f"{BASE_DIR}pwc{PREVIOUS_WORDS_CONSIDERED}-win{WINDOW}-vec{VECTOR_SIZE}-min{MIN_COUNT}"
