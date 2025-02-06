IMAGE_BASE_FOLDER = '../data_source_demo/images'
MASK_BASE_FOLDER = '../data_source_demo/masks'
SAVED_MODELS_FOLDER = '../saved_models'

TEST_SIZE = 0.3
BATCH_SIZE = 32
IMAGE_SIZE = (128,128)

# SNN
NUM_EPOCHS_SNN = 1
PAIR_COUNT = 20             # Number of image pairs to generate during testing/validation for the SNN
SNN_LEARNING_RATE = 0.0001  # 0.001
DISTANCE_THRESHOLD = 0.3    # Distance Threshold

# In debug/test we can visualize the object detected in images
SAVE_MODELS = True


