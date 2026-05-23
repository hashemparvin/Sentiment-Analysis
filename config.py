import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
NUM_SENT = 6
NUM_CHUNK = 6

EPOCHS = 20
LR = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 10
MIN_DELTA = 1e-4
USE_AMP = True

CLASS_NAMES = ["negative","neutral","positive"]
NUM_CLASSES = 3

SAVE_DIR = "outputs"
