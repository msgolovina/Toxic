import torch


HOST = '0.0.0.0'
PORT = '6006'

# device
if torch.cuda.is_available:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# dataset params
CLASS_COLS = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
COMMENT_COL = 'comment_text'

# model path
MODEL_DIR = 'trained_models'
TRAINED_MODEL_PATH = MODEL_DIR + '/pytorch_model.bin'

# model params
BERT_NAME = 'bert-base-cased'
BATCH_SIZE = 32
MAX_LENGTH = 512
NUM_CLASSES = 6

# training params
LR = 2e-5
ADAM_EPSILON = 1e-8
NUM_EPOCHS = 2
NUM_WARMUP_STEPS = 10 ** 3
WEIGHT_DECAY = 0.01
