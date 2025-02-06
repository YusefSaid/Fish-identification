from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os
import random
from torchvision.transforms import functional as F


@dataclass
class FishMaskDataset(Dataset):
    image_mask_pairs: List[Tuple[str, str, int]]  # Include label in the tuple
    transform: Optional[Callable] = None
    _unique_object_ids: Optional[set] = None  # Cache for unique object IDs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path, label = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize and pad
        image, mask = resize_and_pad(image, mask, size=(128, 128))

        mask_array = np.array(mask)

        object_ids = np.unique(mask_array)
        object_ids = object_ids[object_ids != 0]  # Exclude background

        boxes, labels, masks = [], [], []

        for obj_id in object_ids:
            obj_mask = (mask_array == obj_id).astype(np.uint8)
            non_zero_indices = np.argwhere(obj_mask > 0)

            ymin, xmin = non_zero_indices.min(axis=0)
            ymax, xmax = non_zero_indices.max(axis=0)

            if ymax > ymin and xmax > xmin:  # Ensure valid boxes
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(label))  # Use the provided label
                masks.append(obj_mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_path': image_path  # Add image_path here
        }

        return image, target




    @property
    def unique_object_ids(self):
        if self._unique_object_ids is None:
            # Compute unique object IDs
            unique_ids = set()
            for _, mask_path, _ in self.image_mask_pairs:  # Ignore label here
                mask = np.array(Image.open(mask_path).convert("L"))
                unique_ids.update(np.unique(mask))
            unique_ids.discard(0)  # Remove background
            self._unique_object_ids = unique_ids
        return self._unique_object_ids

    @staticmethod
    def create_dataloaders(
            image_base_folder: str,
            mask_base_folder: str,
            transform: Optional[callable] = None,
            batch_size: int = 4,
            test_size: float = 0.2,
            random_seed: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
            max_images_per_class: Optional[int] = None,  # New parameter
    ) -> Tuple[DataLoader, DataLoader, dict[str: int]]:
        """
        Creates train and test DataLoaders for the FishMaskDataset with subfolder-based labels.

        Args:
            image_base_folder (str): Path to the base folder containing image subfolders.
            mask_base_folder (str): Path to the base folder containing mask subfolders.
            transform: Transformations to apply to the images and masks.
            batch_size (int): Batch size for DataLoaders.
            test_size (float): Proportion of data to use for the test set.
            random_seed (Optional[int]): Random seed for reproducibility.
            collate_fn (Optional[Callable]): Custom collate function for DataLoader.
            max_images_per_class (Optional[int]): Maximum number of images per class. Use None for no limit.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
        """
        # Ensure reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Create a mapping of subfolder names to class labels
        subfolders = sorted([subfolder for subfolder in os.listdir(image_base_folder) if
                             os.path.isdir(os.path.join(image_base_folder, subfolder))])

        subfolder_to_label = {subfolder: idx for idx, subfolder in enumerate(subfolders)}
        #subfolder_to_label['background'] = 0 # add this for visual

        print(f"Subfolder to Label Mapping: {subfolder_to_label}")


        # Collect image-mask pairs with labels
        image_mask_pairs = []
        for subfolder, label in subfolder_to_label.items():
            image_subfolder_path = os.path.join(image_base_folder, subfolder)
            mask_subfolder_name = subfolder.replace('fish_', 'mask_')
            mask_subfolder_path = os.path.join(mask_base_folder, mask_subfolder_name)

            # Validate subfolder paths
            if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
                files = [f for f in os.listdir(image_subfolder_path) if f.endswith('.png')]
                # Limit the number of images per class
                if max_images_per_class is not None:
                    files = files[:max_images_per_class]

                for image_file in files:
                    identifier = image_file.replace('fish_', '').replace('.png', '')
                    mask_file = f"mask_{identifier}.png"
                    mask_path = os.path.join(mask_subfolder_path, mask_file)

                    image_path = os.path.join(image_subfolder_path, image_file)
                    if os.path.exists(mask_path):
                        image_mask_pairs.append((image_path, mask_path, label))
                    else:
                        print(f"Warning: Mask not found for image {image_file}")

        # Check if pairs were found
        if not image_mask_pairs:
            raise ValueError("No image-mask pairs found. Verify directory structure and file naming conventions.")

        # Split into training and test sets
        train_pairs, test_pairs = train_test_split(
            image_mask_pairs, test_size=test_size, random_state=random_seed
        )

        # Create datasets with labels included
        train_dataset = FishMaskDataset(
            [(img, mask, lbl) for img, mask, lbl in train_pairs],
            transform=transform
        )
        test_dataset = FishMaskDataset(
            [(img, mask, lbl) for img, mask, lbl in test_pairs],
            transform=transform
        )

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn or FishMaskDataset.collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn or FishMaskDataset.collate_fn
        )

        return train_dataloader, test_dataloader, subfolder_to_label


def resize_and_pad(image, mask, size=(128, 128)):
    """
    Resizes and pads the image and mask to the specified size.

    Args:
        image (PIL.Image): The input image.
        mask (PIL.Image): The input mask.
        size (Tuple[int, int]): The target size (height, width).

    Returns:
        Tuple[PIL.Image, PIL.Image]: Resized image and mask.
    """
    image = F.resize(image, size)
    mask = F.resize(mask, size)
    return image, mask
