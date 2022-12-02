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

class MagicalCNNLSTM(nn.Module):
    def __init__(self,
                input_channels,
                fc_size,
                representation_dim,
                output_size,
                hidden_units,
                num_layers,
                freeze_cnn=True):
        super().__init__()
        self.input_channels = input_channels
        self.fc_size = fc_size
        self.representation_dim = representation_dim
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.freeze_cnn = freeze_cnn

        self.cnn = MAGICALCNN(input_channels=input_channels, fc_size=fc_size, representation_dim=representation_dim)
        if (freeze_cnn):
            print('freezing cnn weights')
            full_model = torch.load('checkpoints/model_pandmagic_lr1e-4.pt')
            for key in [x for x in full_model.keys() if 'extract_features' in x]:
                newkey = key[17:]
                self.cnn.state_dict()[newkey].copy_(full_model[key])
            # self.cnn.load_state_dict(torch.load('checkpoints/model_pandmagic_lr1e-4.pt'))
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            print('not freezing cnn weights')
        self.lstm = ShallowRegressionLSTM(input_size=representation_dim, output_size=output_size, hidden_units=hidden_units, num_layers=num_layers)

    def forward(self, X):
        b, t, c, h, w = X.shape
        # combine the batch and sequence dimensions to pass into cnn first, then split them back out for lstm
        X = torch.reshape(X, (b*t, c, h, w))
        X = self.cnn(X)
        X = torch.reshape(X, (b, t, self.representation_dim))
        out = self.lstm(X)

        return out

class BC_custom(nn.Module):
    def __init__(self, input_size, output_size, net_arch, log_std_init=0, deterministic=False, ortho_init=True, extractor='flatten', freeze_cnn=True):
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
            self.extract_features = MAGICALCNN(input_channels=3, fc_size=128)
        elif (self.extractor == 'magicalcnnlstm'):
            self.extract_features = MagicalCNNLSTM(input_channels=3, fc_size=128, representation_dim=self.input_size,
                                                    output_size=self.input_size, hidden_units=32, num_layers=2, freeze_cnn=freeze_cnn)
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
