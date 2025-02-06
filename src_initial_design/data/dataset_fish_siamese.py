import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import constants
from torch.utils.data import DataLoader

# updated to mathc up with siamese network
@dataclass
class FishSiameseDataset(Dataset):
    pairs: list[tuple[str, str]]
    labels: list[int]
    transform: Optional[Callable] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # Get paths from the dataset
        image_path1, image_path2 = self.pairs[idx]
        label = self.labels[idx]

        # Open images using PIL
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        # Apply transformations, if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Return the pair of images and the label
        return (image1, image2), label




    @staticmethod
    def create_dataloaders(
            image_base_folder: str,
            transform=None,
            batch_size: int = 32,
            test_size: float = 0.2,
            random_seed: int | None = None,
            pair_count: int = 1000  # New parameter to control number of pairs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Creates train and test DataLoaders for the Siamese Network.
        Args:
            image_base_folder (str): Path to the folder containing images.
            transform: Transformations to apply to the images.
            batch_size (int): Batch size for the DataLoader.
            test_size (float): Fraction of pairs to reserve for testing.
            random_seed (int): Seed for reproducibility.
            pair_count (int): Total number of pairs to generate (training + testing).
        Returns:
            Tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
        """
        # Load image paths and labels
        print('Loading images and labels...')
        image_paths, labels = FishSiameseDataset._load_image_paths_and_labels(image_base_folder)

        # Generate limited pairs and labels for Siamese Network
        print(f'Generating pairs...(PAIR COUNT: {pair_count})')
        pairs, pair_labels = FishSiameseDataset._generate_limited_pairs(image_paths, labels, pair_count)

        # Split into training and testing sets
        print('Splitting pairs into train and test sets...')
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            pairs, pair_labels, test_size=test_size, random_state=random_seed
        )

        # Create datasets
        print('Creating datasets...')
        train_dataset = FishSiameseDataset(train_pairs, train_labels, transform=transform)
        test_dataset = FishSiameseDataset(test_pairs, test_labels, transform=transform)

        # Create DataLoaders
        print('Creating DataLoaders...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def _load_image_paths_and_labels(image_base_folder: str):
        image_paths = []
        labels = []

        # Each subfolder in the base folder is a class
        for label, subfolder in enumerate(os.listdir(image_base_folder)):
            class_folder = os.path.join(image_base_folder, subfolder)
            if os.path.isdir(class_folder):
                for image_file in os.listdir(class_folder):
                    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Add other extensions if needed
                        image_paths.append(os.path.join(class_folder, image_file))
                        labels.append(label)

        if not image_paths:
            raise ValueError("No images found in the specified folder.")

        return image_paths, labels

    @staticmethod
    def _generate_limited_pairs(
            image_paths: List[str],
            labels: List[int],
            pair_count: int) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Optimized version to generate a limited number of image pairs for training and testing.
        Args:
            image_paths (List[str]): List of image paths.
            labels (List[int]): List of labels corresponding to the images.
            pair_count (int): Total number of pairs to generate.
        Returns:
            Tuple[List[Tuple[str, str]], List[int]]: List of pairs and their labels.
        """
        pairs = []
        pair_labels = []
        num_images = len(image_paths)

        for _ in range(pair_count):
            # Randomly select two images
            idx1, idx2 = random.sample(range(num_images), 2)

            # Determine similarity label
            label = 1 if labels[idx1] == labels[idx2] else 0

            # Add the pair and label
            pairs.append((image_paths[idx1], image_paths[idx2]))
            pair_labels.append(label)

        return pairs, pair_labels


    @staticmethod
    def create_dataloaders_from_mask_r_cnn(
            pairs,
            pair_labels,
            pairs_test,
            pairs_labels_test,
            transform=None,
            batch_size: int = 32,
            shuffle: bool = True) -> tuple[DataLoader, DataLoader]:

        train_loader = FishSiameseDataset._create_dataloader_from_mask_r_cnn(
            pairs,
            pair_labels,
            transform,
            batch_size,
            shuffle
        )

        test_loader = FishSiameseDataset._create_dataloader_from_mask_r_cnn(
            pairs_test,
            pairs_labels_test,
            transform,
            batch_size,
            shuffle
        )

        return train_loader, test_loader



    @staticmethod
    def _create_dataloader_from_mask_r_cnn(pairs,
                                          pair_labels,
                                          transform=None,
                                          batch_size: int = 32,
                                          shuffle: bool = True):

        train_dataset = FishSiameseDataset(pairs, pair_labels, transform=transform)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


