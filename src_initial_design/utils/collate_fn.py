
def collate_fn(batch):
    """
    Custom collate function to handle batches with varying image sizes.

    Args:
        batch: List of tuples (image, target), where
            - image: Tensor of shape [C, H, W]
            - target: Dictionary of target data (e.g., boxes, labels, masks)

    Returns:
        Tuple[List[Tensor], List[Dict]]: A tuple containing a list of images
        and a list of target dictionaries.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)