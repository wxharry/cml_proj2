import os
import argparse
from typing import Dict
import ray
from ray.air import Checkpoint

from torch import nn

import ray.train as train

from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchPredictor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

my_infr_checkpoint = Checkpoint.from_directory("models/")
my_model = NeuralNetwork()

batch_predictor = BatchPredictor.from_checkpoint(
    my_infr_checkpoint, TorchPredictor, model=my_model
)

data_dir = "cml_proj2/inference/dataset"
ds = ray.data.read_images(data_dir, size=(28, 28)).limit(3)

predicted_probabilities = batch_predictor.predict(ds)
predicted_probabilities.show()