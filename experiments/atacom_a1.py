import torch

from mushroom_rl.utils.torch import TorchUtils

from atacom import VelocityControlSystem
from atacom import Constraint, ConstraintList
from atacom import ATACOMController
from atacom import AgentWrapper

class ATACOMWrapper(AgentWrapper):
    def __init__(self, env_info, atacom_controller: ATACOMController, learning_agent, randomize_dynamics=False):
        
        self.env_info = env_info
        
        super().__init__(atacom_controller=atacom_controller, learning_agent=learning_agent, randomize_dynamics=randomize_dynamics)
        
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']], 0.
 
    def _void_action(self):
        return torch.zeros((self.env_info['num_envs'], self.env_info['action']['len']), device=TorchUtils.get_device())
        
class JointPosConstraint(Constraint):
    def __init__(self, n_joints, joint_limits, logger=None):
        name = 'Joint_pos'
        self.n_joints = n_joints
        self.joint_limits = joint_limits
        self.logger = logger
        super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints * 2, dim_z=0)

    def fun(self, q, z=None):
        # TODO Check if necessary substract defaults joint position 
        pos = q[:, :self.n_joints]
        result = torch.cat([pos - self.joint_limits[1], self.joint_limits[0] - pos], dim=1)
        if self.logger is not None:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        J_pos = torch.vstack([torch.eye(self.n_joints), -torch.eye(self.n_joints)])
        return J_pos.unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device)
    
class FootPosConstraint(Constraint):
    def __init__(self, n_joints, side, get_foot, get_foot_J, alpha=1.5, beta=1.5, min_z=-0.2, max_z=0., logger=None):
        name = side + '_foot_pos'
        self.n_joints = n_joints
        self.logger = logger
        self.get_foot = get_foot
        self.get_foot_J = get_foot_J
        self.alpha = alpha
        self.beta = beta
        self.link_name = side + '_foot'
        self.min_z = min_z
        self.max_z = max_z
        super().__init__(name, dim_q=self.n_joints, dim_k=3, dim_z=0)

    def fun(self, q, z=None):
        # TODO Add tanh and commands
        H = self.get_foot(q, self.link_name)
        pos = H[:, :3, 3]

        xy = pos[:, 0] ** 2 / (self.alpha ** 2) + pos[:, 1] ** 2 / (self.beta ** 2) - 1
        z_high = pos[:, 2] - self.max_z
        z_low = self.min_z - pos[:, 2]
        # print(f'Z max: {pos[:, 2].max()} Z min: {pos[:, 2].min()}')
        
        result = torch.stack([xy, z_high, z_low], dim=1)
        if self.logger is not None:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        # TODO Add tanh and commands
        H = self.get_foot(q, self.link_name)
        J = self.get_foot_J(q, self.link_name)
        pos = H[:, :3, 3]

        J_pos = torch.stack([2 * pos[:, 0] / (self.alpha ** 2), 2 * pos[:, 1] / (self.beta ** 2)], dim=1).unsqueeze(1)
        J_xy = (J_pos @ J[:, :2]).squeeze()
        J_z_high = J[:, 2]
        J_z_low = -J[:, 2]

        return torch.stack([J_xy, J_z_high, J_z_low], dim=1).to(q.device)
        
def build_atacom_agent(rl_agent, env_info, atacom_params):
    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=env_info['n_joints'])
    # constr_list.add_constraint(JointPosConstraint(env_info['robot']['n_joints'], env_info['joint_pos_limit'].to(TorchUtils.get_device()), logger=env_info['logger']))
    
    # No possible action
    limit = torch.tensor([0.2 for _ in range(env_info['n_joints'])], dtype=torch.float32).to(TorchUtils.get_device())
    joint_limit = torch.vstack([-limit , limit])
    #constr_list.add_constraint(JointPosConstraint(env_info['n_joints'], joint_limit, logger=env_info['logger']))
    
    constr_list.add_constraint(FootPosConstraint(env_info['n_joints'], 'FL', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], logger=env_info['logger']))
    constr_list.add_constraint(FootPosConstraint(env_info['n_joints'], 'FR', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], logger=env_info['logger']))
    constr_list.add_constraint(FootPosConstraint(env_info['n_joints'], 'RL', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], logger=env_info['logger']))
    constr_list.add_constraint(FootPosConstraint(env_info['n_joints'], 'RR', env_info['fun']['get_relative_link'], env_info['fun']['get_J_relative_link'], logger=env_info['logger']))
    atacom_controller = ATACOMController(constr_list, dyn,
                                         slack_beta=atacom_params['slack_beta'],
                                         slack_tol=atacom_params['slack_tol'],
                                         slack_dynamics_type=atacom_params['slack_dynamics_type'],
                                         drift_compensation_type=atacom_params['drift_compensation_type'],
                                         drift_clipping=atacom_params['drift_clipping'],
                                         lambda_c=atacom_params['lambda_c'])
    return ATACOMWrapper(env_info=env_info,
                         atacom_controller=atacom_controller,
                         learning_agent=rl_agent,
                         randomize_dynamics=atacom_params['randomize_dynamics'])


