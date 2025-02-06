import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), # activation function to introduce non-linearity
            nn.MaxPool2d(2, 2), # Downsamples the features maps
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512), # input size matching output size of CNN
            nn.ReLU(), # activation function to introduce non-linearity
            nn.Linear(512, 256) #Reduces the final feature vector size to 256
        )

    # Purpose: Processes a single input through the CNN and FC layers.
    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    # Purpose: Compares two inputs (input1 and input2) by computing the distance between their feature vectors.
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Calculate the Euclidean distance between the two feature vectors
        return torch.norm(output1 - output2, dim=1)