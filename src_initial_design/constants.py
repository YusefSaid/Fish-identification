IMAGE_BASE_FOLDER = '../data_source_demo/images'
MASK_BASE_FOLDER = '../data_source_demo/masks'
SAVED_MODELS_FOLDER = '../saved_models'

TEST_SIZE = 0.2
BATCH_SIZE = 10
IMAGE_SIZE = (128,128)

# SNN
NUM_EPOCHS_SNN = 1
PAIR_COUNT = 500            # Number of image pairs to generate during testing/validation for the SNN
SNN_LEARNING_RATE = 0.0001  # 0.001

# Mask-RCNN
NUM_EPOCHS_MRCNN = 1
MAX_IMAGES_PER_CLASS = 10        #  Maximum number of images to load per class for training/testing the Mask R-CNN.
RCNN_CONFIDENCE_THRESHOLD = 0.5   #  Minimum confidence score required for a detection to be considered valid
RCNN_LEARNING_RATE = 0.0001

# In debug/test we can visualize the object detected in images
VISUALIZE_PREDICTIONS = False
SAVE_MODELS = False


