from typing import Callable
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import constants
from data.dataset_fish_siamese import ReferenceFishDataset
from PIL import Image


def evaluate_snn(model,
                 test_loader: DataLoader,
                 criterion: Callable,
                 device: torch.device):
    """
    Evaluates a Siamese Neural Network using a fixed threshold from constants.THRESHOLD.

    Args:
        model (nn.Module): The trained Siamese Neural Network model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (Callable): Loss function (e.g., ContrastiveLoss).
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average loss and accuracy on the test set.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    threshold = constants.DISTANCE_THRESHOLD  # Fixed threshold for evaluation
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        count = 0
        for (images1, images2), labels in test_loader:
            # Move data to the device
            count+=1
            images1, images2, labels = images1.to(device), images2.to(device), labels.float().to(device)

            # Forward pass
            distances = model(images1, images2)  # Compute similarity (distances)

            # Compute loss
            loss = criterion(distances, labels)
            total_loss += loss.item()
            num_batches += 1

            # Make predictions based on the fixed threshold
            predictions = (distances < threshold).float()  # 1 if distance < threshold, else 0
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            print(f'\rAccuracy: {100 * correct_predictions/total_samples:.2f}%. (Distance Threshold={threshold:.2f})'
                  f' ({count}/{len(test_loader)})', end='')

    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = (correct_predictions / total_samples) * 100

    # Log results
    print(f"Evaluation using Fixed Threshold ({threshold:.2f}):")
    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def test_image(path_to_image: str,
               image_reference_base_folder: str,
               model,
               transform,
               device,
               batch_size=32):
    """
    Test a query image against a reference dataset using a trained Siamese Neural Network.

    Args:
        path_to_image (str): Path to the query image.
        model (nn.Module): Trained Siamese Neural Network model.
        transform: Transform to apply to the images.
        device (torch.device): Device to run the model ('cpu' or 'cuda').
        batch_size (int): Batch size for the reference DataLoader.

    Returns:
        int: Predicted label for the query image.
    """
    # Load the query image and apply the transform
    query_image = Image.open(path_to_image).convert("RGB")
    if transform:
        query_image = transform(query_image)
    query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension

    # Create a reference DataLoader with one image per class
    reference_loader = ReferenceFishDataset.create_reference_dataloader(
        image_base_folder=image_reference_base_folder,
        transform=transform,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    best_match = None
    highest_similarity = float("-inf")

    # Iterate through the reference DataLoader
    with torch.no_grad():
        for images, labels in reference_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Pass query image and reference images through the SNN
            distances = model(query_image.expand_as(images), images)  # Compare query with batch

            # Find the most similar image
            batch_highest_similarity, best_match_idx = distances.min(0)  # Get closest distance
            if batch_highest_similarity > highest_similarity:
                highest_similarity = batch_highest_similarity
                best_match = labels[best_match_idx].item()  # Get corresponding label


    return best_match, highest_similarity
