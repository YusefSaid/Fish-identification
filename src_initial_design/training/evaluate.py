from typing import Callable
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate_snn(model: nn.Module,
                 test_loader: DataLoader,
                 criterion: Callable,
                 device: torch.device):
    """
    Evaluates a Siamese Neural Network on the test set.

    Args:
        model (nn.Module): The Siamese Neural Network model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (Callable): Loss function (e.g., BCEWithLogitsLoss).
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        avg_loss (float): Average loss on the test set.
        accuracy (float): Accuracy percentage on the test set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        for (images1, images2), labels in test_loader:
            # Move data to the GPU/CPU
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.float().to(device)

            # Forward pass
            similarity = model(images1, images2)

            # Compute loss
            loss = criterion(similarity, labels)
            total_loss += loss.item()
            num_batches += 1

            # Classification accuracy based on a threshold (e.g., 0.5 for BCE)
            predictions = (similarity >= 0.5).float()  # 1: similar, 0: dissimilar
            correct_predictions += (predictions == labels).sum().item()

            # Debugging: Print predictions and labels if needed
            # print(f"Predictions: {predictions}")
            # print(f"Labels: {labels}")

    # Compute average loss and accuracy
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / len(test_loader.dataset) * 100

    print(f"Average Loss on Test Set: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

