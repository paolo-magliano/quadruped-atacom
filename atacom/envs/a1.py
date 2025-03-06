import torch
import pinocchio as pin

from mushroom_rl.environments.isaacsim_envs.isaac_a1_pos_action import A1Pos 
from mushroom_rl.environments.isaacsim_envs.isaac_a1_vel_action import A1Vel
from mushroom_rl.environments.isaacsim_envs.isaac_a1_legged_gym import IsaacA1Description as A1Eff
from mushroom_rl.utils.isaac_sim import ObservationType
from mushroom_rl.rl_utils.spaces import Box

from atacom.envs.costr_log_utils import ConstrLogger

class A1EffVel(A1Eff):
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_position=(40, 0, 4), camera_target=(30, 0, 0)):
        super().__init__(num_envs, horizon, headless, domain_randomization, camera_position, camera_target)
        self._action_scale = 1.5
        action_limit = self._task.get_joint_max_velocities() / self._action_scale
        self._mdp_info.action_space = Box(-action_limit, action_limit, data_type=action_limit.dtype)

        self._integral_error = torch.zeros((num_envs, action_limit.shape[0]), device=self._device)

    def _compute_torque(self, action, joint_vels, joint_pos):
        self._torques = 1. * (self._action_scale * action - joint_vels) + 0.1 * self._integral_error

        not_sat = self._torques.abs() < self._effort_limit

        self._torques = self._torques.clamp(-self._effort_limit, self._effort_limit)
        self._integral_error[not_sat] += self._action_scale * action[not_sat] - joint_vels[not_sat]

        return self._torques
    
    def reset_all(self, env_mask, state=None):
        self._integral_error[env_mask] = 0.
        return super().reset_all(env_mask, state)

class A1Env(A1Vel):
    def __init__(self, cfg):
        super().__init__(cfg['num_envs'], cfg['horizon'], cfg['headless']) 
        if cfg['urdf_filepath']:
            self.urdf_filepath = cfg['urdf_filepath']
        else:
            self.urdf_filepath = 'atacom/envs/assets/a1.urdf'

        self._model = pin.buildModelFromUrdf(self.urdf_filepath)
        self._model_data = [self._model.createData() for _ in range(cfg['num_envs'])]
        default_q = self._default_joint_angles.clone().repeat(cfg['num_envs'], 1)
        self._last_q = torch.zeros_like(default_q)
        self._update_model_data(default_q)

        self._leg_base_H ={frame.split('_')[0]: H_matrix(torch.eye(3), self._model_data[0].oMf[self._model.getFrameId(frame)].translation) for frame in ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh']}
        self._leg_base_Ad = {k: Ad_matrix(v) for k, v in self._leg_base_H.items()}

        self.constraints_logger = ConstrLogger(cfg['num_envs'])

        self.env_info = dict()
        self.env_info['num_envs'] = cfg['num_envs']

        self.env_info['n_joints'] = self._model.nq
        self.env_info['joint_pos_limit'] = dict()
        self.env_info['joint_vel_limit'] = dict()
        self.env_info['default_joint_pos'] = self._default_joint_angles
        self.env_info['joint_pos_limit'] = torch.vstack([torch.tensor(self._model.lowerPositionLimit, dtype=torch.float32), torch.tensor(self._model.upperPositionLimit, dtype=torch.float32)])
        self.env_info['joint_vel_limit'] = torch.vstack([-torch.tensor(self._model.velocityLimit, dtype=torch.float32), torch.tensor(self._model.velocityLimit, dtype=torch.float32)])
        
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
        self.env_info['fun']['get_relative_link'] = self.get_relative_link
        self.env_info['fun']['get_J_relative_link'] = self.get_J_relative_link
        self.env_info['fun']['get_commands'] = self.get_commands

        self.env_info['logger'] = self.constraints_logger

        self.env_info['urdf_path'] = self.urdf_filepath

    @classmethod
    def build_env(cls, cfg):
        env = cls(cfg)
        return env, env.env_info
    
    def get_relative_link(self, q, link_name):
        self._update_model_data(q)
        pos = torch.stack([self._relative_link(self._model_data[i], link_name) for i in range(q.shape[0])]).type(q.dtype).to(q.device)
        return pos
    
    def get_J_relative_link(self, q, link_name):
        self._update_model_data(q)
        J = torch.stack([self._J_relative_link(q[i], self._model_data[i], link_name) for i in range(q.shape[0])]).type(q.dtype).to(q.device)
        return J
    
    def _update_model_data(self, q):
        if not torch.allclose(q.double(), self._last_q.double()):
            self._last_q = q.clone()
            for i in range(q.shape[0]):
                pin.forwardKinematics(self._model, self._model_data[i], q[i].detach().cpu().numpy())
                pin.updateFramePlacements(self._model, self._model_data[i])
      
    def _relative_link(self, data, link_name):
        assert link_name.split('_')[0] in self._leg_base_H
        link_idx = self._model.getFrameId(link_name)
        Rl = data.oMf[link_idx].rotation
        tl = data.oMf[link_idx].translation
        Hl = H_matrix(Rl, tl)
        Hb = self._leg_base_H[link_name.split('_')[0]]

        return torch.matmul(torch.inverse(Hb), Hl)
    
    def _J_relative_link(self, q, data, link_name):
        assert link_name.split('_')[0] in self._leg_base_Ad
        link_idx = self._model.getFrameId(link_name)
        
        Jl = pin.computeFrameJacobian(self._model, data,  q.detach().cpu().numpy(), link_idx, pin.LOCAL_WORLD_ALIGNED)

        # Ad_b = self._leg_base_Ad[link_name.split('_')[0]]

        #TODO Check with finite difference if torch.inverse(Ad_b) or Ad_b
        return torch.tensor(Jl) # torch.matmul(torch.inverse(Ad_b), torch.tensor(Jl))
    
    def get_commands(self):
        return self.commands.clone()

    def step_all(self, env_mask, action):
        obs, reward, done, info = super().step_all(env_mask, action)
        costr_info = self.constraints_logger.get_and_reset()
        assert self.constraints_logger.empty()
        # for i in range(len(info)):
        #     info[i].update(costr_info[i].copy())
        return obs, reward, done, costr_info.copy()

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
