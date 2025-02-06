from models.siamese_network import *
from transforms.custom_transforms import CustomTransform
from training.evaluate import  test_image
from utils.model_io import load_model
import os
import constants

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
print("Loading model ...")
model = load_model(model,
                   os.path.join(constants.SAVED_MODELS_FOLDER,"20241202_173430_snn")
                   , device )

def run_image_test(img_path: str):
    print(f"Test image: '{img_path}'")
    best_match, highest_similarity = test_image(img_path,
            constants.IMAGE_BASE_FOLDER,
            model,
            CustomTransform(),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"), 32)

    print(f"fish_01: Predicted label: {best_match} with similarity: {highest_similarity:.4f}")

    return best_match, highest_similarity


if __name__ == '__main__':

    image_path = os.path.join('../data_source_min/images/fish_01/fish_000000009598_05281.png')
    run_image_test(image_path)

    image_path = os.path.join('../data_source_min/images/fish_02/fish_000000249596_03656.png')
    run_image_test(image_path)


