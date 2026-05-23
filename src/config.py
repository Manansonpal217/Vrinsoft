"""Central configuration for the customer-support reply assistant.

Every tunable knob lives here so the preprocessing, training, inference and
serving code all agree on the same hyper-parameters and file locations.
"""
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")
RAW_CSV = os.path.join(DATA_DIR, "twcs", "twcs.csv")          # full TWCS dataset
SAMPLE_CSV = os.path.join(DATA_DIR, "sample.csv")             # 100-row preview
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

PAIRS_CSV = os.path.join(ARTIFACTS_DIR, "pairs.csv")          # cleaned query/reply pairs
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
MODEL_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "model_config.json")
ENCODER_WEIGHTS = os.path.join(ARTIFACTS_DIR, "encoder.weights.h5")
DECODER_WEIGHTS = os.path.join(ARTIFACTS_DIR, "decoder.weights.h5")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "history.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "eval_metrics.json")

# --------------------------------------------------------------------------
# Special tokens  (index order is fixed - do not reorder)
# --------------------------------------------------------------------------
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# Data / preprocessing
# --------------------------------------------------------------------------
MAX_PAIRS = 80_000          # number of (query, reply) pairs to keep for training
MAX_INPUT_LEN = 32          # max tokens in a customer message (encoder)
MAX_OUTPUT_LEN = 32         # max tokens in a support reply (decoder)
MIN_TOKENS = 2              # drop pairs whose query/reply is shorter than this
VOCAB_SIZE = 12_000         # keep the N most frequent tokens (+ 4 special tokens)
RANDOM_SEED = 42
VAL_FRACTION = 0.05
TEST_FRACTION = 0.05

# --------------------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------------------
EMBEDDING_DIM = 128
UNITS = 256                 # GRU hidden units (encoder & decoder)
DROPOUT = 0.2

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1.0e-3
CLIPNORM = 5.0

# --------------------------------------------------------------------------
# Decoding / inference
# --------------------------------------------------------------------------
DEFAULT_DECODING = "greedy"   # "greedy" or "sample"
TEMPERATURE = 0.7
TOP_K = 10

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
