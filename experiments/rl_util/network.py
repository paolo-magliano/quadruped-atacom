import torch
import torch.nn as nn


ActivationLayer = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'selu': nn.SELU,
    'leaky_relu': nn.LeakyReLU,
    'softplus': nn.Softplus,
}

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation, gain_coeff=1., **kwargs):
        super().__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=gain_coeff*nn.init.calculate_gain(activation))
            self.model.append(layer)
            self.model.append(ActivationLayer[activation]())

        self.model.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(self.model[-1].weight,
                                gain=gain_coeff*nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        state = state.float()
        return self.model(torch.squeeze(state, 1))
