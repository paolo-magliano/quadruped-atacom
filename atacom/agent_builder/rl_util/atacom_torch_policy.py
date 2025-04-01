from mushroom_rl.policy.torch_policy import GaussianTorchPolicy, TorchPolicy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from itertools import chain

import torch  
class AtacomGaussianTorchPolicy(TorchPolicy):
    def __init__(self, network, input_shape, output_shape, mean_n_features, std_n_features, std_0=1., policy_state_shape=None, atacom_controller=None, **params):
        super().__init__(policy_state_shape)

        self._action_dim = output_shape[0]

        mu_params = params.copy()
        mu_params['n_features'] = mean_n_features
        self._mu = Regressor(TorchApproximator, input_shape, output_shape, network=network, **mu_params)
        self._predict_params = dict()
        params['gain_coeff'] = 1e-3

        std_params = params.copy()
        std_params['n_features'] = std_n_features
        self._log_sigma = Regressor(TorchApproximator, input_shape, output_shape, network=network, **std_params)
        self.init_log_sigma = torch.log(torch.tensor(std_0))

        self.atacom_controller = atacom_controller  

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _log_sigma='mushroom',
        )

    def draw_action_t(self, state):
        return self.distribution_t(state).sample().detach()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]
    
    def distribution_t(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def get_mean_and_chol(self, state):
        log_sigma = self._log_sigma(state, **self._predict_params) + self.init_log_sigma
        assert torch.all(torch.exp(log_sigma) > 0)
        return self._mu(state, **self._predict_params), torch.diag_embed(torch.exp(log_sigma))

    def set_weights(self, weights):
        self._log_sigma.set_weights(weights[-self._action_dim:])

        self._mu.set_weights(weights[:-self._action_dim])

    def get_weights(self):
        mu_weights = self._mu.get_weights()
        sigma_weights = self._log_sigma.get_weights()

        return torch.concatenate([mu_weights, sigma_weights])

    def parameters(self):
        return chain(self._mu.model.network.parameters(), self._log_sigma.model.network.parameters())

    def set_atacom_controller(self, atacom_controller, env_info):
        self.atacom_controller = atacom_controller
        self.env_info = env_info

    def entropy_t(self, state=None):
        action, _ = self.draw_action(state)
        log_p = self.log_prob_t(state, action)
        # if self.atacom_controller is not None:
        #     q_x, x_dot = self._unwrap_state(state)
        #     _, B = self.atacom_controller.compose_action(q_x, action, x_dot)

        #     _, N, M = B.size()
        #     K = min(N, M)

        #     delta = torch.eye(K) * 1e-4
        #     log_det_B = torch.logdet(B[:, :K, :K] + delta.to(B)).unsqueeze(1)
        #     log_p -= log_det_B
        return - log_p.mean()
    
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.