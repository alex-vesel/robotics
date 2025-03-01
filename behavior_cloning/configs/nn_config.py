TRAIN_EPOCHS = 1000
LOG_STEPS = 10
SAVE_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64

CHUNK_SIZE = 5

# Fully connected angle network config
ANGLE_FC_INPUT_DIM = 7
ANGLE_FC_HIDDEN_DIM = 128
ANGLE_FC_NUM_LAYERS = 4
ANGLE_FC_OUTPUT_DIM = 128

# Neck config
NECK_HIDDEN_DIM = 512
NECK_NUM_LAYERS = 4
NECK_OUTPUT_DIM = 512

RELOAD_MODEL = False

if RELOAD_MODEL:
    SAVE_EPOCHS = 40
