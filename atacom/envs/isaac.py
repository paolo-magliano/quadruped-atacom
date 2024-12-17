import torch
from gym import spaces as gym_spaces

from omni.isaac.kit import SimulationApp
from omniisaacgymenvs.utils.task_util import initialize_task

from mushroom_rl.core import VectorizedEnvironment, MDPInfo
from mushroom_rl.utils.viewer import ImageViewer
from mushroom_rl.utils.isaac_utils import convert_task_observation
from mushroom_rl.rl_utils.spaces import *

from omni.isaac.gym.vec_env import VecEnvBase

class IsaacEnv(ectorizedEnvironment, VecEnvBase):
    def set_task(self, task, backend="torch", sim_params=None, init_sim=True, rendering_dt=1 / 60):
        super().set_task(task, backend, sim_params, init_sim, rendering_dt)
    
        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def stop(self):
        self.close()

    def seed(self, seed=-1):
        from omni.isaac.core.utils.torch.maths import set_seed

        return set_seed(seed)

    def reset_all(self, env_mask, state=None):
        idxs = torch.argwhere(env_mask).squeeze()  # .cpu().numpy()  # takes torch datatype 
        self._task.reset_idx(idxs)
        action = torch.zeros((len(idxs), self._task.action_space.shape[0]), device=env_mask.device)
        observation, _, _, info = self.step_all(env_mask, action)

        return observation, info

    def step_all(self, env_mask, action):
        if self._task.randomize_actions:
            action = self._task._dr_randomizer.apply_actions_randomization(actions=action, reset_buf=self._task.reset_buf)

        action = torch.clamp(action, -self._task.clip_actions, self._task.clip_actions)

        self._task.pre_physics_step(action)

        if (self.sim_frame_count + self._task.control_frequency_inv) % self._task.rendering_interval == 0:
            for _ in range(self._task.control_frequency_inv - 1):
                self._world.step(render=False)
                self.sim_frame_count += 1
            self._world.step(render=self._render)
            self.sim_frame_count += 1
        else:
            for _ in range(self._task.control_frequency_inv):
                self._world.step(render=False)
                self.sim_frame_count += 1

        observation, reward, done, info = self._task.post_physics_step()

        if self._task.randomize_observations:
            observation = self._task._dr_randomizer.apply_observations_randomization(observations=observation, reset_buf=self._task.reset_buf)

        observation = convert_task_observation(observation)
        observation = torch.clamp(observation, -self._task.clip_obs, self._task.clip_obs)

        env_mask_cuda = torch.tensor(env_mask, device=done.device)
        
        return observation.clone(), reward, torch.logical_and(done, env_mask_cuda), [info]*self._n_envs
    
    def render_all(self, env_mask, record=False):
        self._world.render()
        task_render = self._task.get_render()

        self._viewer.display(task_render)

        if record:
            return task_render
    
