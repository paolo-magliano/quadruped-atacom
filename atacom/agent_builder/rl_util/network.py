import torch
import torch.nn as nn


ActivationLayer = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'selu': nn.SELU,
    'leaky_relu': nn.LeakyReLU,
    'softplus': nn.Softplus
}


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation, embedding_size=0, dropout_ratio=0, layer_norm=False, **kwargs):
        super().__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        if embedding_size > 0:
            self.register_buffer('embedding_vec', torch.arange(0, embedding_size, 1).float())
            self.embedding = nn.Linear(1 + embedding_size * 2, 1)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain(activation))
            self.model.append(layer)
            self.model.append(ActivationLayer[activation]())
            if dropout_ratio > 0:
                self.model.append(nn.Dropout(dropout_ratio))
            if layer_norm:
                self.model.append(nn.LayerNorm(n_features[i + 1]))

        self.model.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(
            self.model[-1].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=-1)
        if hasattr(self, 'embedding_vec'):
            x = torch.pi * self.embedding_vec * state_action.unsqueeze(-1)
            state_action = self.embedding(
                torch.cat([state_action.unsqueeze(-1), torch.cos(x), torch.sin(x)], dim=-1)).squeeze(-1)

        q = self.model(state_action)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation, embedding_size=0, **kwargs):
        super().__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        if embedding_size > 0:
            self.register_buffer('embedding_vec', torch.arange(0, embedding_size, 1).float())
            self.embedding = nn.Linear(1 + embedding_size * 2, 1)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain(activation))
            self.model.append(layer)
            self.model.append(ActivationLayer[activation]())

        self.model.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(self.model[-1].weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        state = state.float()
        if hasattr(self, 'embedding_vec'):
            x = torch.pi * self.embedding_vec * state.unsqueeze(-1)
            state = self.embedding(
                torch.cat([state.unsqueeze(-1), torch.cos(x), torch.sin(x)], dim=-1)).squeeze(-1)

        return self.model(torch.squeeze(state, 1))


class QuantileCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation, embedding_size,
                 dropout_ratio=0, layer_norm=False, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        self.base_net = nn.Sequential()
        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
            self.base_net.append(layer)
            if dropout_ratio > 0:
                self.base_net.append(nn.Dropout(dropout_ratio))
            if layer_norm:
                self.base_net.append(nn.LayerNorm(n_features[i + 1]))
            self.base_net.append(ActivationLayer[activation]())

        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_size, n_features[-2]), nn.Sigmoid())  # Sigmoid used in DSAC
        self.register_buffer('embed_vec', torch.arange(0, embedding_size, 1).float())

        self.quantile_net = nn.Sequential()
        self.quantile_net.append(nn.Linear(n_features[-2], n_features[-2]))
        nn.init.xavier_uniform_(self.quantile_net[-1].weight, gain=nn.init.calculate_gain(activation))
        if layer_norm:
            self.quantile_net.append(nn.LayerNorm(n_features[-2]))
        self.quantile_net.append(ActivationLayer[activation]())
        self.quantile_net.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(self.quantile_net[-1].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action, tau):
        assert tau.dim() == 2 and tau.shape[0] == state.shape[0]
        state_action = torch.cat((state.float(), action.float()), dim=-1)  # (B, S + A)
        state_action_embedding = self.base_net(state_action)  # (B, F)
        tau_embedding = self.embedding_net(torch.cos(torch.pi * self.embed_vec *
                                           tau.float().unsqueeze(-1)))  # (B, T, F)

        assert state_action_embedding.shape[0] == tau_embedding.shape[0]
        assert state_action_embedding.shape[1] == tau_embedding.shape[2]
        state_action_embedding = state_action_embedding.unsqueeze(1)  # (B, T, F)

        quantiles = self.quantile_net(state_action_embedding * tau_embedding)  # (B, T, 1)
        return quantiles.squeeze(-1)
