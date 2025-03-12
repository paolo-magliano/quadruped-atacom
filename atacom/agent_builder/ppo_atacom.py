import torch
import matplotlib.pyplot as plt 

from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo_nikita import NikitaPPO
from mushroom_rl.rl_utils.value_functions import compute_gae
from mushroom_rl.utils.minibatches import minibatch_generator

class AtacomPPO(NikitaPPO):

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
                 critic_fit_params=None, clip_grad_norm=1., schedule='adaptive', desired_kl=0.01, atacom_enable=False):
        super().__init__(mdp_info, policy, actor_optimizer, critic_params,
            n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff,
            critic_fit_params, clip_grad_norm, schedule, desired_kl)
        self._atacom_enable = atacom_enable

        self._loss_clip = []
        self._loss_entropy = []

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, last = dataset.parse(to='torch')
        state, next_state, state_old = self._preprocess_state(state, next_state)

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

        self._update_policy(state, action, adv, old_log_p, state, old_pol_dist)
        self._plot_loss()

        # Print fit information
        self._log_info(dataset, state, v_target, old_pol_dist)
        self._iter += 1        

    def _update_policy(self, obs, act, adv, old_log_p, state, old_pol_dist):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                with torch.inference_mode():
                    new_pol_dist = self.policy.distribution_t(state)
                    kl = torch.mean(torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist))
                    self._adapt_learning_rate(kl)

                self._optimizer.zero_grad()
                prob_ratio = torch.exp(self.policy.log_prob_t(obs_i, act_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss_clip = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss_entropy -= self._ent_coeff() * self.policy.entropy_t(obs_i)
                self._loss_clip.append(loss_clip.item())
                self._loss_entropy.append(loss_entropy.item())
                loss = loss_clip + loss_entropy

                loss.backward()
                self._clip_gradient()
                self._optimizer.step()

    def _plot_loss(self):
        if self._iter % 10 != 0:
            return 
        plt.plot(self._loss_clip)
        plt.plot(self._loss_entropy)
        plt.savefig(f'plot/loss/loss_{self._iter}.png')
        plt.close()

        self._loss_clip = []
        self._loss_entropy = []