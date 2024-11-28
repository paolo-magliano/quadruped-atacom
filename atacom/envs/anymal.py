import urdfpy
import torch

from mushroom_rl.environments.isaac_env import IsaacEnv

JOINT_MAP = [
    'LF_HAA',
    'LH_HAA',
    'RF_HAA',
    'RH_HAA',
    'LF_HFE',
    'LH_HFE',
    'RF_HFE',
    'RH_HFE',
    'LF_KFE',
    'LH_KFE',
    'RF_KFE',
    'RH_KFE'
]

class AnymalEnv(IsaacEnv):
    def __init__(self, cfg=None, headless=False, backend='torch'):
        assert cfg['task_name'] == 'Anymal'
        super().__init__(cfg, headless, backend)

        self.dt = self._task.dt

        #Read urdf file
        self.urdf_filepath = 'atacom/envs/anymal/urdf/anymal.urdf'
        self.robot = urdfpy.URDF.load(self.urdf_filepath)

        self.env_info = dict()
        self.env_info['robot'] = dict()
        self.env_info['robot']['n_joints'] = len(self.robot.actuated_joints)
        self.env_info['robot']['joint_pos_limit'] = dict()
        self.env_info['robot']['joint_vel_limit'] = dict()
        for joint in self.robot.actuated_joints:
            self.env_info['robot']['joint_pos_limit'][joint.name] = [joint.limit.lower, joint.limit.upper]
            self.env_info['robot']['joint_vel_limit'][joint.name] = [-joint.limit.velocity, joint.limit.velocity]
        
        self.env_info['robot']['joint_pos_limit'] = torch.tensor([self.env_info['robot']['joint_pos_limit'][joint] for joint in JOINT_MAP]).T
        self.env_info['robot']['joint_vel_limit'] = torch.tensor([self.env_info['robot']['joint_vel_limit'][joint] for joint in JOINT_MAP]).T

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

    @classmethod
    def build_env(cls, cfg, headless=False, backend='torch'):
        env = cls(cfg, headless, backend)
        dynamic_info = env.env_info['robot']
        dynamic_info['env_info'] = env.env_info
        return env, dynamic_info
    