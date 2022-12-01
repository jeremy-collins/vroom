import torch
import torch.nn as nn
import torch.distributions as distributions
from functools import partial
import numpy as np

from lstm import ShallowRegressionLSTM
from cnns import MAGICALCNN

class BC_MLP(nn.Module):
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
            layers.append(nn.Tanh())
            prev_layer_size = layer

        self.layers = nn.Sequential(*layers).to(self.device)

    def forward(self, X):
        return self.layers(X)

class BC_CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(30976, self.output_size)
        )

    def forward(self, X):
        out = self.layers(X)

        return out

class BC_custom(nn.Module):
    def __init__(self, input_size, output_size, net_arch, log_std_init=0, deterministic=False, ortho_init=True, extractor='flatten'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net_arch = net_arch
        self.deterministic = deterministic
        self.ortho_init = ortho_init
        self.extractor = extractor

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if (self.extractor == 'cnn'):
            self.extract_features = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            # freeze everything but last layer
            for param in self.extract_features.parameters():
                param.requires_grad = False
            self.extract_features.fc = nn.Linear(2048, self.input_size)
            for param in self.extract_features.fc.parameters():
                param.requires_grad = True
        elif (self.extractor == 'cnn2'):
            self.extract_features = BC_CNN(self.input_size)
        elif (self.extractor == 'magicalcnn'):
            self.extract_features = MAGICALCNN(input_channels=3)
        elif (self.extractor == 'lstm'):
            self.extract_features = ShallowRegressionLSTM(self.input_size, self.input_size, 32, 2)
        elif (self.extractor == 'flatten'):
            self.extract_features = nn.Flatten() # this can be a CNN for images
        else: 
            print('extractor {} not recognized'.format(self.extractor))

        self.action_net = nn.Linear(net_arch[-1], self.output_size)
        self.value_net = nn.Linear(net_arch[-1], 1)

        self.mlp = BC_MLP(input_size, output_size, net_arch)

        self.mean_actions = nn.Linear(net_arch[-1], output_size)
        self.log_std = nn.Parameter(torch.ones(output_size)*log_std_init, requires_grad=True)

        # initialization
        if (self.ortho_init):
            module_gains = {
                self.extract_features: np.sqrt(2),
                self.mlp: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def forward(self, X):
        features = self.extract_features(X)
        latent = self.mlp(features)
        values = self.value_net(latent)
        mean_actions = self.action_net(latent)
        self.proba_distribution(mean_actions, self.log_std) # self._get_action_dist_from_latent(latent_pi)
        if (self.deterministic):
            # mode
            actions = self.distribution.mean()
        else:
            # random sample
            actions = self.distribution.rsample()
        log_prob = self.sum_independent_dims(self.distribution.log_prob(actions))
        # actions = actions.reshape((-1,) + self.output_size) # not sure what this does
        return actions, values, log_prob

    def proba_distribution(self, mean_actions, log_std):
        action_std = torch.ones_like(mean_actions)*log_std.exp()
        self.distribution = distributions.Normal(mean_actions, action_std)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent = self.mlp(features)
        mean_actions = self.action_net(latent)
        self.proba_distribution(mean_actions, self.log_std) # self._get_action_dist_from_latent(latent_pi)
        log_prob = self.sum_independent_dims(self.distribution.log_prob(actions))
        values = self.value_net(latent)
        entropy = self.sum_independent_dims(self.distribution.entropy())
        return values, log_prob, entropy

    def sum_independent_dims(self, tensor):
        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.

        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
