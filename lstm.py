from torch import nn
import numpy as np
import torch

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=output_size)

    def forward(self, X):
        batch_size = X.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)

        _, (hn, _) = self.lstm(X, (h0, c0))
        out = self.linear(hn[0])

        return out