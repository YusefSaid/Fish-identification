import os

import torch
import torch.nn as nn
import constants

# Save the model's state_dict after training
def save_model(model: nn.Module,
               name: str):
    try:
        if not os.path.exists(constants.SAVED_MODELS_FOLDER):
            os.makedirs(constants.SAVED_MODELS_FOLDER)

        save_path = os.path.join(constants.SAVED_MODELS_FOLDER, name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    except IOError as io_error:
        print(io_error)

def load_model(model: nn.Module,
               name: str,
               device: torch.device) -> nn.Module:
    try:
        load_path = os.path.join(constants.SAVED_MODELS_FOLDER, name)
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)
        return model
    except FileNotFoundError as file_not_found_error:
        print(file_not_found_error)