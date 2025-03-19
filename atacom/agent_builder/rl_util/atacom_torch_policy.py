from mushroom_rl.policy.torch_policy import GaussianTorchPolicy
import torch  
class AtacomGaussianTorchPolicy(GaussianTorchPolicy):
    def __init__(self, network, input_shape, output_shape, std_0=1., policy_state_shape=None, atacom_controller=None, **params):
        super().__init__(network, input_shape, output_shape, std_0, policy_state_shape, **params)
        self.atacom_controller = atacom_controller  

    def set_atacom_controller(self, atacom_controller, env_info):
        self.atacom_controller = atacom_controller
        self.env_info = env_info

    def entropy_t(self, state=None):
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
        return - log_p.mean()
    
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.
