import torch
import torch.nn as nn
import torch.distributions as distributions
from functools import partial
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, net_arch):
        super().__init__()
        self.input_size = input_size
        self.net_arch = net_arch
        self.output_size = output_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        layers = []
        prev_layer_size = input_size
        for layer in net_arch:
            layers.append(nn.Linear(prev_layer_size, layer))
            layers.append(nn.ReLU())
            prev_layer_size = layer

        self.layers = nn.Sequential(*layers).to(self.device)
        self.encode = nn.Linear(net_arch[-1], self.output_size)

    def forward(self, X):
        out = self.layers(X)
        out = self.encode(out)
        return out