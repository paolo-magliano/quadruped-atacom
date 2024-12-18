import torch
import pinocchio as pin

from mushroom_rl.environments.isaac_env import IsaacEnv
from atacom.envs.anymal_log_utils import AnymalConstrLogger

class AnymalEnv(IsaacEnv):
    def __init__(self, cfg=None, headless=False, backend='torch'):
        assert cfg['task_name'] == 'Anymal'
        super().__init__(cfg, headless, backend)

        self.dt = self._task.dt

        if cfg['urdf_filepath']:
            self.urdf_filepath = cfg['urdf_filepath']
        else:
            self.urdf_filepath = 'atacom/envs/urdf/anymal.urdf'

        self._model = pin.buildModelFromUrdf(self.urdf_filepath)
        self._model_data = [self._model.createData() for _ in range(cfg['num_envs'])]
        self._last_q = torch.zeros_like(self._task._anymals.get_joint_positions())
        self._update_model_data(self._task._anymals.get_joint_positions().clone())

        self._leg_base_H ={frame.split('_')[0]: H_matrix(torch.eye(3), self._model_data[0].oMf[self._model.getFrameId(frame)].translation) for frame in ['LF_HIP', 'LH_HIP', 'RF_HIP', 'RH_HIP']}
        self._leg_base_Ad = {k: Ad_matrix(v) for k, v in self._leg_base_H.items()}

        self.constraints_logger = AnymalConstrLogger(cfg['num_envs'])

        # Repalce limits with self._task._anymals
        # num_dof
        # _dof_indices
        # _dof_names
        # get_dof_limits()

        # _link_indices - 1

        # self.commands_x
        # self.commands_y
        # self.commands_yaw

        self.env_info = dict()
        self.env_info['robot'] = dict()
        self.env_info['robot']['n_joints'] = self._model.nq
        self.env_info['robot']['joint_pos_limit'] = dict()
        self.env_info['robot']['joint_vel_limit'] = dict()

        self.env_info['robot']['joint_pos_limit'] = torch.vstack([torch.tensor(self._model.lowerPositionLimit, dtype=torch.float32), torch.tensor(self._model.upperPositionLimit, dtype=torch.float32)])
        self.env_info['robot']['joint_vel_limit'] = torch.vstack([-torch.tensor(self._model.velocityLimit, dtype=torch.float32), torch.tensor(self._model.velocityLimit, dtype=torch.float32)])
        self.env_info['action_space'] = dict()
        self.env_info['action_space']['num'] = self.info.action_space.shape[0]
        self.env_info['action_space']['limit'] = self.info.action_space
        self.env_info['observation_space'] = dict()
        self.env_info['observation_space']['num'] = self.info.observation_space.shape[0]
        self.env_info['observation_space']['limit'] = self.info.observation_space
        self.env_info['observation_space']['base_lin_vel_idx'] = [0, 1, 2]
        self.env_info['observation_space']['base_ang_vel_idx'] = [3, 4, 5]
        self.env_info['observation_space']['projected_gravity_idx'] = [6, 7, 8]
        self.env_info['observation_space']['commands_scaled_idx'] = [9, 10, 11]
        self.env_info['observation_space']['joint_pos_idx'] = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.env_info['observation_space']['joint_vel_idx'] = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        self.env_info['observation_space']['action_idx'] = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

        self.env_info['num_envs'] = cfg['num_envs']

        self.env_info['function'] = dict()
        self.env_info['function']['get_relative_link'] = self.get_relative_link
        self.env_info['function']['get_J_relative_link'] = self.get_J_relative_link

        self.env_info['logger'] = self.constraints_logger

    @classmethod
    def build_env(cls, cfg, headless=False, backend='torch'):
        env = cls(cfg, headless, backend)
        dynamic_info = env.env_info['robot']
        dynamic_info['env_info'] = env.env_info
        return env, dynamic_info
    
    def get_relative_link(self, q, link_name):
        self._update_model_data(q)
        pos = torch.stack([self._relative_link(data, link_name) for data in self._model_data]).type(q.dtype).to(q.device)
        return pos
    
    def get_J_relative_link(self, q, link_name):
        self._update_model_data(q)
        J = torch.stack([self._J_relative_link(q[i], self._model_data[i], link_name) for i in range(len(self._model_data))]).type(q.dtype).to(q.device)
        return J
    
    def _update_model_data(self, q):
        assert q.shape[0] == len(self._model_data)
        if not torch.allclose(q, self._last_q):
            self._last_q = q.clone()
            for i in range(len(self._model_data)):
                pin.forwardKinematics(self._model, self._model_data[i], q[i].detach().cpu().numpy())
                pin.updateFramePlacements(self._model, self._model_data[i])
        
    def _relative_link(self, data, link_name):
        assert link_name.split('_')[0] in self._leg_base_H
        assert data.oMf[self._model.getFrameId('anymal')] == data.oMf[self._model.getFrameId('universe')]
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

        Ad_b = self._leg_base_Ad[link_name.split('_')[0]]

        #TODO Check wih finite difference if torch.inverse(Ad_b) or Ad_b
        return torch.matmul(torch.inverse(Ad_b), torch.tensor(Jl))
    
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
