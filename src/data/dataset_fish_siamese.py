from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass, field
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple, List
import constants


@dataclass
class FishSiameseDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        """
        Args:
            pairs: List of tuples (image1, image2).
            labels: List of labels (1 for similar, 0 for dissimilar).
            transform: Optional transformation to apply to the images.
        """
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label

    @staticmethod
    def create_dataloaders(
            image_base_folder: str,
            transform=None,
            batch_size: int = 4,
            test_size: float = constants.TEST_SIZE,
            random_seed: int | None = None,
            pair_count: int | None = None  # Maximum number of pairs per class
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Creates train and test DataLoaders for the FishDataset with pairs.

        Args:
            image_base_folder (str): Path to the base folder containing image subfolders.
            transform: Transform to apply to the images and masks.
            batch_size (int): Batch size for DataLoaders.
            test_size (float): Proportion of data to use for the test set.
            random_seed (int): Random seed for reproducibility.
            pair_count (int): Maximum number of pairs per class (optional).

        Returns:
            Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
        """
        # Set seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Dictionary to hold images by subfolder
        subfolder_to_images = {}

        # Collect images from each subfolder
        for subfolder in os.listdir(image_base_folder):
            if subfolder.startswith('fish_'):  # Ensure we only process valid fish subfolders
                subfolder_path = os.path.join(image_base_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    images = [
                        os.path.join(subfolder_path, f)
                        for f in os.listdir(subfolder_path)
                        if f.endswith('.png')
                    ]
                    if images:
                        subfolder_to_images[subfolder] = images

        # Create pairs and labels
        pairs = []
        labels = []

        # Create same-class pairs (label = 1)
        for subfolder, images in subfolder_to_images.items():
            same_class_pairs = []
            for i in range(len(images)):
                for j in range(i + 1, len(images)):  # Pair each image with the others
                    same_class_pairs.append((images[i], images[j]))

            # Restrict to `pair_count` pairs per class if specified
            if pair_count is not None and len(same_class_pairs) > pair_count:
                same_class_pairs = random.sample(same_class_pairs, pair_count)

            pairs.extend(same_class_pairs)
            labels.extend([1] * len(same_class_pairs))

        # Create different-class pairs (label = 0)
        subfolders = list(subfolder_to_images.keys())
        for i in range(len(subfolders)):
            for j in range(i + 1, len(subfolders)):  # Pair images from different subfolders
                images1 = subfolder_to_images[subfolders[i]]
                images2 = subfolder_to_images[subfolders[j]]

                different_class_pairs = [
                    (img1, img2) for img1 in images1 for img2 in images2
                ]

                # Restrict to `pair_count` pairs per class if specified
                if pair_count is not None and len(different_class_pairs) > pair_count:
                    different_class_pairs = random.sample(different_class_pairs, pair_count)

                pairs.extend(different_class_pairs)
                labels.extend([0] * len(different_class_pairs))

        # Check if any pairs were found
        if not pairs:
            raise ValueError("No pairs found. Check the directory structure and file extensions.")

        # Split data into training and test sets
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            pairs,
            labels,
            test_size=test_size,
            random_state=random_seed
        )

        # Create Dataset objects
        train_dataset = FishSiameseDataset(train_pairs, train_labels, transform=transform)
        test_dataset = FishSiameseDataset(test_pairs, test_labels, transform=transform)

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader


class ReferenceFishDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def create_reference_dataloader(
        image_base_folder: str,
        transform=None,
        batch_size: int = 4,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader containing only one image per class.

        Args:
            image_base_folder (str): Path to the base folder with class subfolders.
            transform: Transformations to apply to the images.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the DataLoader.

        Returns:
            DataLoader: Reference DataLoader.
        """
        image_paths = []
        labels = []
        class_to_image = {}

        # Collect one image per class
        for class_idx, class_name in enumerate(sorted(os.listdir(image_base_folder))):
            class_path = os.path.join(image_base_folder, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.endswith(".png") or image_name.endswith(".jpg"):
                        image_path = os.path.join(class_path, image_name)
                        class_to_image[class_idx] = image_path
                        break  # Take only one image per class

        # Prepare the dataset
        for class_idx, image_path in class_to_image.items():
            image_paths.append(image_path)
            labels.append(class_idx)

        dataset = ReferenceFishDataset(image_paths, labels, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


