import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import PPO
from mushroom_rl.rl_utils.value_functions import compute_gae

class AtacomPPO(PPO):

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
            n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
            critic_fit_params=None, atacom_enable=True):
        super().__init__(mdp_info, policy, actor_optimizer, critic_params,
            n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff,
            critic_fit_params)
        self._atacom_enable = atacom_enable
        

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, last = dataset.parse(to='torch')
        state, next_state, state_old = self._preprocess_state(state, next_state)

        # Use the next policy state to retrive the original rl action
        if self._atacom_enable:
            _ , action = dataset.parse_policy_state(to='torch')

        v_target, adv = compute_gae(self._V, state, next_state, reward, absorbing, last, self.mdp_info.gamma,
                                    self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        adv = adv.detach()
        v_target = v_target.detach()

        old_pol_dist = self.policy.distribution_t(state_old)
        old_log_p = old_pol_dist.log_prob(action)[:, None].detach()

        self._V.fit(state, v_target, **self._critic_fit_params)

        self._update_policy(state, action, adv, old_log_p)

        # Print fit information
        self._log_info(dataset, state, v_target, old_pol_dist)
        self._iter += 1

