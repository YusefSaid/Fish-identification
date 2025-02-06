import torch
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass, field
from typing import Tuple, Callable
import constants

import torch
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass, field
from typing import Tuple, Callable
import constants

@dataclass
class CustomTransform:
    image_size: Tuple[int, int] = constants.IMAGE_SIZE
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    transform: Callable = field(init=False)

    def __post_init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(), # Random horizontal flip increases variety
            # adjust brightness, contrast, saturation, hue
            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                   saturation=self.saturation, hue=self.hue),
            transforms.ToTensor(),

            # ImageNet standardization
            # Normalization helps get data within a range and reduces the skewness which helps learn faster and better
            # https://pytorch.org/vision/stable/models.html
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)



