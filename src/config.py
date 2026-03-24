"""Project-wide configuration constants."""

DATA_ROOT = "img-train"
SPLIT_CSV = "data/splits/split.csv"
MODEL_PATH = "models/best_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
