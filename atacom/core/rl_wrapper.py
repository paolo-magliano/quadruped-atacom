import torch
from copy import deepcopy
from .atacom_controller import ATACOMController
from mushroom_rl.core import Agent

from types import SimpleNamespace

class AgentWrapper(Agent):
    def __init__(self, atacom_controller: ATACOMController, learning_agent, randomize_dynamics=False):
        self.atacom_controller = atacom_controller
        self.learning_agent = learning_agent

        self.randomize_dynamics = randomize_dynamics
        super().__init__(self.learning_agent.mdp_info, SimpleNamespace(policy_state_shape=self.learning_agent.mdp_info.action_space.shape), backend='torch')

    def draw_action(self, state_orig, policy_state=None):
        rl_action, _ = self.learning_agent.draw_action(self.learning_agent_preprocess(state_orig.clone()), policy_state)
        low = self.mdp_info.action_space.low if type(self.mdp_info.action_space.low) == torch.Tensor else torch.tensor(self.mdp_info.action_space.low, device=rl_action.device)
        high = self.mdp_info.action_space.high if type(self.mdp_info.action_space.high) == torch.Tensor else torch.tensor(self.mdp_info.action_space.high, device=rl_action.device)
        sampled_action = torch.clip(rl_action, low, high)

        q_x, x_dot = self._unwrap_state(state_orig.clone())
        actual_action = self.atacom_controller.compose_action(q_x, sampled_action.clone(), x_dot)
        actual_action = torch.clip(actual_action, low, high)
        
        # Use the next policy state to return and save the original rl action
        return actual_action, rl_action

    def episode_start(self, initial_state, episode_info):
        if self.randomize_dynamics:
            self.atacom_controller.system_dynamics.randomize()

        _, current_theta = self.learning_agent.episode_start(initial_state, episode_info)

        void_action = self._void_action()

        return void_action, current_theta

    def _unwrap_state(self, state):
        raise NotImplementedError
    
    def _void_action(self):
        return NotImplementedError

    def fit(self, dataset):
        processed_dataset = self.process_dataset_before_fit(dataset)
        self.learning_agent.fit(processed_dataset)

    def learning_agent_preprocess(self, state):

        for p in self.learning_agent._agent_preprocessors:
            if state.ndim == 2:
                state = torch.tensor([p(s.clone()) for s in state])
            else:
                state = p(state.clone())
        return state

    def process_dataset_before_fit(self, dataset):
        dataset_new = deepcopy(dataset)
        dataset_new._data._states = dataset.array_backend.from_list([self.learning_agent_preprocess(s) for s in dataset.state])
        dataset_new._data._next_states = dataset.array_backend.from_list([self.learning_agent_preprocess(s) for s in dataset.next_state])
        return dataset_new

    def stop(self):
        self.learning_agent.stop()

