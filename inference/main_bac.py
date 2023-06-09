import os
import argparse
from typing import Dict
import ray
from ray.data import read_images, from_torch
from ray.air import Checkpoint

import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F

from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchPredictor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

my_infr_checkpoint = Checkpoint.from_directory("models/")
my_model = NeuralNetwork()

batch_predictor = BatchPredictor.from_checkpoint(
    my_infr_checkpoint, TorchPredictor, model=my_model
)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset = datasets.MNIST(
    root="~/data",
    train=False,
    download=True,
    transform=transform,
)
ds = from_torch(dataset)

# data_dir = "cml_proj2/inference/dataset"
# ds = read_images(data_dir, size=(28, 28), include_paths=True)
# predicted_probabilities = batch_predictor.predict(ds, feature_columns=["image"])

predicted_probabilities = batch_predictor.predict(ds)
predicted_probabilities.show()
