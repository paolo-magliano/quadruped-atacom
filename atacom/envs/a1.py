import torch
from pathlib import Path
import itertools

from mushroom_rl.environments.isaacsim_envs.isaac_a1_pos_action import A1Pos 
from mushroom_rl.environments.isaacsim_envs.isaac_a1_vel_action import A1Vel
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description as A1Eff
from mushroom_rl.utils.isaac_sim import ObservationType
from mushroom_rl.rl_utils.spaces import Box

from atacom.envs.costr_log_utils import ConstrLogger
from atacom.util.extended_state_observer import Observer

from experiments.util.plotter import Plotter, StoreData

class A1EffVel(A1Eff):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_position=(105, 0, 4), camera_target=(95, 0, 0), action_scale=1.5, Kp=1., Kd=0., Ki=0.1):
        super().__init__(num_envs, horizon, headless, domain_randomization, camera_position, camera_target)
        self.last_dof_pos = torch.zeros((num_envs, self.NUM_DOFS), device=self._device)
        self._action_scale = action_scale
        self._set_controller_gains(Kp, Kd, Ki)
        action_limit = self._task.get_joint_max_velocities() / self._action_scale
        self._mdp_info.action_space = Box(-action_limit, action_limit, data_type=action_limit.dtype)

        self._last_joint_vel = torch.zeros((num_envs, self.NUM_DOFS), device=self._device)
        self._integral_error = torch.zeros((num_envs, self.NUM_DOFS), device=self._device)

        self.action_ratio = torch.ones((num_envs, self.NUM_DOFS), device=self._device)

        self.plotter = Plotter(data_dim=2, n_row=4, n_col=3, title="pi_controller", path="plot/controller", data_labels=["target_action", "actual_action"])
        self.controller_data = StoreData(data_dim=2, n_row=4, n_col=3, num_envs=num_envs)

        state_gains = [50., 50.]
        self.disturbance_gains = [1000., 500.]

        self.observers = [Observer(num_envs, self.NUM_DOFS, self.dt, self._device, s_g, d_g) for s_g, d_g in zip(state_gains, self.disturbance_gains)]

        self.observer_plotter = Plotter(data_dim=len(self.observers) + 1, data_len=100, n_row=4, n_col=3, title="observer", path="plot/observer", data_labels=["state", *[f'sg_{s_g}_dg_{d_g}' for s_g, d_g in zip(state_gains, self.disturbance_gains)]])
        self.disturbance_plotter = Plotter(data_dim=len(self.observers) + 1, data_len=50, n_row=4, n_col=3, title="disturbance", path="plot/observer", data_labels=["velocity", *[f'sg_{s_g}_dg_{d_g}' for s_g, d_g in zip(state_gains, self.disturbance_gains)]])


    def _set_controller_gains(self, Kp, Kd, Ki):
        self._Kp = self._reapeat_dof(Kp)
        self._Kd = self._reapeat_dof(Kd)
        self._Ki = self._reapeat_dof(Ki)

    def _reapeat_dof(self, value):
        if self.NUM_DOFS % len(value) != 0:
            raise ValueError(f"Invalid gains size: {len(value)}")
        repeat_len = self.NUM_DOFS // len(value)
        return torch.tensor(value).repeat(repeat_len).to(self._device)

    def _compute_torque(self, action, joint_vels, joint_pos):
        # action = torch.zeros_like(action)
        # action = action - self.observers[0].get_disturbance_estimate() / self.disturbance_gains[0] * 100 
        self._torques = self._Kp * (self._action_scale * action - joint_vels) + self._Kd * (self._last_joint_vel - joint_vels) +  self._Ki * self._integral_error

        not_sat = self._torques.abs() < self._effort_limit

        self._torques = self._torques.clamp(-self._effort_limit, self._effort_limit)
        self._last_joint_vel = joint_vels.clone().detach()
        self._integral_error[not_sat] += self._action_scale * action[not_sat] - joint_vels[not_sat]

        return self._torques
    
    def reset_all(self, env_mask, state=None):
        self._integral_error[env_mask] = 0.
        for observer in self.observers:
            observer.reset()
        return super().reset_all(env_mask, state)
    
    def step_all(self, env_mask, action):
        obs, reward, done, info = super().step_all(env_mask, action)
        ob_state = obs[:, self.observation_helper.obs_idx_map["joint_pos"]]
        ob_vel = obs[:, self.observation_helper.obs_idx_map["joint_vel"]]

        # plot_state = torch.stack([ob_state, *[observer.update_estimate(ob_state, action)[0] for observer in self.observers]], dim=1)
        # self.observer_plotter.add_data(plot_state[0])
        # plot_vel = torch.stack([ob_vel, *[observer.get_disturbance_estimate() / self.disturbance_gains[i] * 100 for i, observer in enumerate(self.observers)]], dim=1)
        # self.disturbance_plotter.add_data(plot_vel[0])

        return obs, reward, done, info
    
    # def _step_finalize(self, env_indices):
    #     super()._step_finalize(env_indices)
    #     self._integral_error = torch.zeros_like(self._integral_error)

    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        dof_vel = self.observation_helper.get_from_obs(next_obs, "joint_vel")
        dof_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")

        target_pos = dof_pos + self._action_scale * action * self.dt
        # self.controller_data.add_data(action * self._action_scale, dof_vel)
        # self.plotter.add_data(action[0] * self._action_scale, dof_vel[0])

        #---------------------------------------------------------------------------

        r_tracking_lin_vel = self._reward_tracking_lin_vel(base_lin_vel_xy) * 1.0 * self.dt
        r_tracking_ang_vel = self._reward_tracking_ang_vel(base_ang_vel_z) * 0.5 * self.dt
        r_lin_vel_z = self._reward_lin_vel_z(base_lin_vel_z) * -2.0 * self.dt
        r_ang_vel_xy = self._reward_ang_vel_xy(base_ang_vel_xy) * -0.05 * self.dt
        r_torques = self._reward_torques(self._torques) * -0.0002 * self.dt
        r_dof_acc = self._reward_dof_acc(dof_vel) * -2.5e-7 * self.dt
        r_feet_air_time = self._reward_feet_air_time() * 1.0 * self.dt
        r_collision = self._reward_collision() * -1. * self.dt
        r_action_rate = self._reward_action_rate(action) * -0.01 * self.dt
        r_dof_pos_rate = self._reward_dof_pos_rate(target_pos) * -0.01 * self.dt
        r_dof_pos_limits = self._reward_dof_pos_limits(dof_pos) * -10.0 * self.dt

        self._extra_info_rewards = {
            "tracking_lin_vel": r_tracking_lin_vel, "tracking_ang_vel": r_tracking_ang_vel,
            "lin_vel_z": r_lin_vel_z, "ang_vel_xy": r_ang_vel_xy, 
            "torques": r_torques, "dof_acc": r_dof_acc, 
            "feet_air_time": r_feet_air_time, "collision": r_collision, 
            "action_rate": r_action_rate, 
            "dof_pos_rate": r_dof_pos_rate, "dof_pos_limits": r_dof_pos_limits
        }

        reward = r_tracking_lin_vel + r_tracking_ang_vel + r_lin_vel_z + r_ang_vel_xy + r_torques + r_dof_acc + r_feet_air_time \
                + r_collision + r_dof_pos_rate + r_dof_pos_limits + r_action_rate
        
        reward = torch.clamp(reward, min=0.)

        # reward = 0.1 - 5 * self.dt * self._reward_default_dof_pos(dof_pos) - 5 * self.dt * self._reward_action_ratio() 

        self.last_actions = action.clone().detach()
        self.last_dof_vel = dof_vel.clone().detach()
        self.last_dof_pos = target_pos.clone().detach()
        
        return reward

    def _reward_dof_pos_rate(self, dof_pos):
        return torch.sum(torch.square(self.last_dof_pos - dof_pos), dim=1)

    def _reward_default_dof_pos(self, dof_pos):
        return torch.sum(torch.square(self._default_joint_angles - dof_pos), dim=1)

    # def _reward_action_ratio(self):
    #     return torch.sum(torch.square(self.action_ratio - 1.), dim=1)

    # def step_all(self, env_mask, action):
    #     real_action, action_ratio = action
    #     self.action_ratio = action_ratio
    #     return super().step_all(env_mask, real_action)


class A1Atacom():
    def __init__(self, cfg):
        if cfg['urdf_filepath']:
            self.urdf_filepath = cfg['urdf_filepath']
        else:
            self.urdf_filepath = str(Path(__file__).resolve().parent / 'assets/a1.urdf')

        self.constraints_logger = ConstrLogger()

        self.env_info = dict()
        self.env_info['num_envs'] = cfg['num_envs']

        self.env_info['n_joints'] = self.NUM_DOFS
        self.env_info['joint_pos_limit'] = dict()
        self.env_info['joint_vel_limit'] = dict()
        self.env_info['default_joint_pos'] = self._default_joint_angles
        self.env_info['joint_pos_limit'] = self._task.get_joint_pos_limits()
        self.env_info['joint_vel_limit'] = torch.vstack([-self._task.get_joint_max_velocities(), self._task.get_joint_max_velocities()])
        
        self.env_info['action'] = dict()
        self.env_info['action']['len'] = self._mdp_info.action_space.shape[0]
        self.env_info['action']['limit'] = self._mdp_info.action_space
        self.env_info['action']['idx'] = {s: [i for i, name in enumerate(self._action_spec) if s in name] for s in ['FL', 'FR', 'RL', 'RR']}

        self.env_info['obs'] = dict()
        self.env_info['obs']['len'] = self._mdp_info.observation_space.shape[0]
        self.env_info['obs']['limit'] = self._mdp_info.observation_space
        self.env_info['obs']['base_lin_vel_idx'] = self.observation_helper.obs_idx_map["base_lin_vel"]
        self.env_info['obs']['base_ang_vel_idx'] = self.observation_helper.obs_idx_map["base_lin_vel"]
        self.env_info['obs']['projected_gravity_idx'] = self.observation_helper.obs_idx_map["projected_gravity"]
        self.env_info['obs']['commands_idx'] = self.observation_helper.obs_idx_map["commands"]
        self.env_info['obs']['joint_pos_idx'] = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_POS]
        self.env_info['obs']['joint_vel_idx'] = self.observation_helper.obs_types_idx_map[ObservationType.JOINT_VEL]
        self.env_info['obs']['actions_idx'] = self.observation_helper.obs_idx_map["actions"]

        self.env_info['fun'] = dict()
        self.env_info['fun']['get_commands'] = self.get_commands

        self.env_info['logger'] = self.constraints_logger

        self.env_info['urdf_path'] = self.urdf_filepath

    @classmethod
    def build_env(cls, cfg):
        env = cls(cfg)
        return env, env.env_info  
    
    def get_commands(self):
        return self.commands.clone()

    def step_all(self, env_mask, action):
        obs, reward, done, info = super().step_all(env_mask, action)
        costr_info = self.constraints_logger.get_and_reset()
        assert self.constraints_logger.empty()
        info.update(costr_info)
        return obs, reward, done, info
    
class A1PIEnv(A1Atacom, A1EffVel):
    def __init__(self, cfg):
        A1EffVel.__init__(self, cfg['num_envs'], cfg['horizon'], cfg['headless'], action_scale=cfg['control']['action_scale'], Kp=cfg['control']['Kp'], Kd=cfg['control']['Kd'], Ki=cfg['control']['Ki'])
        A1Atacom.__init__(self, cfg)

class A1PDEnv(A1Atacom, A1Vel):
    def __init__(self, cfg):
        A1Vel.__init__(self, cfg['num_envs'], cfg['horizon'], cfg['headless'], action_scale=cfg['control']['action_scale'], stiffness=cfg['control']['Kp'], damping=cfg['control']['Kd'], integral=cfg['control']['Ki'])
        A1Atacom.__init__(self, cfg)

def H_matrix(R, t):
    R = torch.tensor(R) if not isinstance(R, torch.Tensor) else R
    t = torch.tensor(t) if not isinstance(t, torch.Tensor) else t
    return torch.cat([torch.cat([R, t.unsqueeze(1)], dim=1), torch.tensor([0., 0., 0., 1.]).unsqueeze(0)], dim=0)

def Ad_matrix(H):
    H = torch.tensor(H) if not isinstance(H, torch.Tensor) else H
    R = H[:3, :3]
    t = H[:3, 3]
    t_R = skew_matrix(t) @  R
    Ad = torch.cat([torch.cat([R, t_R], dim=1), torch.cat([torch.zeros(3, 3), R], dim=1)], dim=0)
    return Ad

def skew_matrix(v):
    return torch.tensor([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
