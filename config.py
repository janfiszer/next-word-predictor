from enum import Enum


class NeuralNetworkArchitectureType(Enum):
    FNN = 0
    RNN = 1


# architecture
NN_TYPE = NeuralNetworkArchitectureType.RNN

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
PREVIOUS_WORDS_CONSIDERED = 5
END_TOKEN = '[END]'

# for the NN
if NN_TYPE == NeuralNetworkArchitectureType.RNN:
    INPUT_SIZE = (PREVIOUS_WORDS_CONSIDERED, VECTOR_SIZE)
elif NN_TYPE == NeuralNetworkArchitectureType.FNN:
    INPUT_SIZE = (PREVIOUS_WORDS_CONSIDERED * VECTOR_SIZE, )

# training parameters
EPOCHS = 16
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LABEL_SMOOTHING_RATIO = 0.2

# directories of trained models
BASE_DIR = "trained_models"
# NN_MODEL_DIR = f"{MODELS_DIR}/dense/min{VOCAB_MIN_COUNT}-pwc{PREVIOUS_WORDS_CONSIDERED}-lr{LEARNING_RATE}-batch{BATCH_SIZE}/"
MODEL_DIR = f"{BASE_DIR}/{NN_TYPE.name}/lstm/min{VOCAB_MIN_COUNT}-pwc{PREVIOUS_WORDS_CONSIDERED}-lr{LEARNING_RATE}-batch{BATCH_SIZE}/"
WORD2VEC_DIR = f"{BASE_DIR}/word2vecs/"
