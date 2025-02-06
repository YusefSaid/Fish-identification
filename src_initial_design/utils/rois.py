import itertools
from typing import Any
import constants
from torchvision import transforms
from torch._C.cpp import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
import torch
import random

# region of intrest
def extract_rois(model: nn.Module, dataloader: DataLoader, device) -> list[dict]:
    model.eval()
    roi_data = []  # List to store ROIs and their corresponding labels

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            # Get predictions from the Mask R-CNN
            predictions = model(images)

            for idx, prediction in enumerate(predictions):
                boxes = prediction["boxes"]  # Detected bounding boxes
                labels = prediction["labels"]  # Detected labels
                scores = prediction["scores"]  # Confidence scores

                # Filter detections by a confidence threshold
                high_conf_indices = scores > constants.RCNN_CONFIDENCE_THRESHOLD
                boxes = boxes[high_conf_indices]
                labels = labels[high_conf_indices]

                # Extract ROIs
                for box, label in zip(boxes, labels):
                    xmin, ymin, xmax, ymax = box.int().tolist()
                    roi = crop(images[idx].cpu(), ymin, xmin, ymax - ymin, xmax - xmin)
                    roi_data.append({
                        "roi": roi,  # Extracted ROI tensor
                        "label": label.item(),  # Class label
                        "image_path": targets[idx]["image_path"]  # Access image_path
                    })

    return roi_data



def generate_pairs_from_rois(roi_data: list[dict],
                             pair_count: int) -> tuple[list[tuple[str, str]], list[int]]:
    """
    Generate pairs of ROIs and their corresponding labels.

    Args:
        roi_data (list[dict]): List of ROIs and their metadata.
        pair_count (int): Number of pairs to generate.

    Returns:
        Tuple[List[Tuple[str, str]], List[int]]: List of pairs and their labels.
    """
    pairs = []
    pair_labels = []

    for i, data1 in enumerate(roi_data):
        for j, data2 in enumerate(roi_data):
            if i != j:  # Avoid pairing the same ROI
                roi1_path = data1["image_path"]
                roi2_path = data2["image_path"]
                label1 = data1["label"]
                label2 = data2["label"]

                pairs.append((roi1_path, roi2_path))
                pair_labels.append(1 if label1 == label2 else 0)  # 1 if same class, else 0

                if len(pairs) >= pair_count:
                    return pairs, pair_labels

    return pairs, pair_labels


def generate_pairs_for_snn(model: nn.Module,
                           dataloader_train: DataLoader,
                           dataloader_test: DataLoader,
                           device):
    print("Extracting ROIs from images ...")
    roi_data_training = extract_rois(
        model,
        dataloader_train,
        device)

    # Generate pairs from ROIs
    pairs, pair_labels = generate_pairs_from_rois(
        roi_data_training,
        pair_count=constants.PAIR_COUNT
    )

    roi_data_test = extract_rois(
        model,
        dataloader_test,
        device)

    # Generate pairs from ROIs
    pairs_test, pair_labels_test = generate_pairs_from_rois(
        roi_data_test,
        pair_count=constants.PAIR_COUNT
    )

    # Validate before creating the DataLoader
    if len(pairs) == 0 or len(pair_labels) == 0 or len(pairs_test) == 0 or len(pair_labels_test) == 0:
        raise ValueError("No pairs or labels generated. Check ROI extraction and pair generation.")

    return pairs, pair_labels, pairs_test, pair_labels_test

