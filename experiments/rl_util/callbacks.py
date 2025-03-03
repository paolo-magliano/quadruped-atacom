import wandb

from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.core.dataset import VectorizedDataset

from rl_util.compute_metric import get_metrics
from experiments.util.log_info import log_info

class LogDataset(Callback):
    def __init__(self, agent, atacom_enable, gamma, logger, n_episodes):
        self.agent = agent
        self.atacom_enable = atacom_enable
        self.gamma = gamma
        self.logger = logger
        self.dataset = None
        self.n_episodes = n_episodes
    
    def __call__(self, dataset):
        # TODO Check dataset._info to be reduced to n_episodes
        policy_state = dataset._data.policy_state[:, :self.n_episodes].clone() if dataset._data.is_stateful else None
        policy_state_next = dataset._data.next_policy_state[:, :self.n_episodes].clone() if dataset._data.is_stateful else None
        dataset = VectorizedDataset.from_array(dataset._data.state[:, :self.n_episodes].clone(), dataset._data.action[:, :self.n_episodes].clone(), dataset._data.reward[:, :self.n_episodes].clone(), dataset._data.next_state[:, :self.n_episodes].clone(), dataset._data.absorbing[:, :self.n_episodes].clone(), dataset._data.last[:, :self.n_episodes].clone(),
                                                policy_state, policy_state_next, dataset._data.mask[:, :self.n_episodes].clone() if dataset._data.mask is not None else None, dataset._info, dataset._episode_info, dataset._theta_list[:self.n_episodes], dataset._dataset_info.horizon, dataset._dataset_info.gamma, dataset._dataset_info.backend, dataset._dataset_info.device, n_envs=self.n_episodes)
        if self.dataset is None:
            self.dataset = dataset
        else:
            self.dataset += dataset
        if len(self.dataset) >= 1000:
            self.dataset = self.dataset[:1000]
            info = self.get()
            self.log(*info)
            self.dataset = None

    
    def clean(self):
        self.dataset.clear()

    def get(self):
        return get_metrics(self.dataset.flatten(), self.agent, self.atacom_enable, self.gamma)
    
    def log(self, *info):
        log_dict = log_info(self.logger, self.agent, *info)
        wandb.log(log_dict)

        