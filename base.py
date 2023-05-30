# Defining variables and configurations
import torch

MODEL_NAME = 'vesteinn/ScandiBERT-no-faroese' 
PATH_TO_DATA = 'data/faroese/Faroese.csv'
#OUTPUT_DIR='./results

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 2
MAX_TOKENS = 514
NUMBER_EXAMPLES = 2000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MASK_PROBABILITY =  0.15

