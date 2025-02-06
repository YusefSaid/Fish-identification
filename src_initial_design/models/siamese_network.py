import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Shared CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers to produce embeddings
        self.shared_fc = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),  # Adjust input size (flattened CNN output)
            nn.ReLU(),
            nn.Linear(512, 256),  # Embedding size of 256
            nn.ReLU()
        )

    def forward_once(self, x):
        """
        Pass a single input through the network.
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        embedding = self.shared_fc(x)
        return embedding

    def forward(self, input1, input2):
        """
        Compute embeddings for both inputs and return their similarity.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute Euclidean distance between embeddings
        distance = torch.norm(output1 - output2, dim=1)
        return distance
