import torch  

from mushroom_rl.policy.torch_policy import GaussianTorchPolicy
from mushroom_rl.utils.numerical_gradient import numerical_diff_function

class AtacomGaussianTorchPolicy(GaussianTorchPolicy):
    def __init__(self, network, input_shape, output_shape, std_0=1., policy_state_shape=None, atacom_controller=None, **params):
        super().__init__(network, input_shape, output_shape, std_0, policy_state_shape, **params)
        self.atacom_controller = atacom_controller  

    def set_atacom_controller(self, atacom_controller, env_info):
        self.atacom_controller = atacom_controller
        self.env_info = env_info

    def entropy_t(self, state=None):
        entropy = self._entropy_t(state)
        auto_grad = self._check_entropy_gradient(state)
        entropy.backward()
        grad = self._log_sigma.grad
        return entropy

    def _entropy_t(self, state=None):
        action, _ = self.draw_action(state)
        log_p = self.log_prob_t(state, action)
        if self.atacom_controller is not None:
            q_x, x_dot = self._unwrap_state(state)
            _, B = self.atacom_controller.compose_action(q_x, action, x_dot)

            _, N, M = B.size()
            K = min(N, M)

            delta = torch.eye(K) * 1e-4
            log_det_B = torch.logdet(B[:, :K, :K] + delta.to(B)).unsqueeze(1)
            log_p -= log_det_B
        return -log_p.mean()
    
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.
    

    def _check_entropy_gradient(self, state=None):
        def fun(log_sigma):
            self._log_sigma = torch.tensor(log_sigma, dtype=self._log_sigma.dtype, device=self._log_sigma.device)
            return self._entropy_t(state).cpu().detach().numpy()
        
        log_sigma = self._log_sigma.cpu().detach().numpy()
        grad = torch.tensor(numerical_diff_function(fun, log_sigma, eps=1e-5)).to(self._log_sigma.device)
        return grad
