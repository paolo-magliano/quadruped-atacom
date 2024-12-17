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
        return obs[:, self.env_info['observation_space']['joint_pos_idx']], 0.
 
    def _void_action(self):
        return torch.zeros((self.env_info['num_envs'], self.env_info['action_space']['num']), device=TorchUtils.get_device())
        
class JointPosConstraint(Constraint):
    def __init__(self, n_joints, joint_limits, logger=None):
        name = 'joint_pos'
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
    def __init__(self, n_joints, side, get_foot, get_foot_J, alpha=10.5, beta=10.5, min_z=-0.3, max_z=-0.1, logger=None):
        name = side + '_foot_pos'
        self.n_joints = n_joints
        self.logger = logger
        self.get_foot = get_foot
        self.get_foot_J = get_foot_J
        self.alpha = alpha
        self.beta = beta
        self.link_name = side + '_SHANK'
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
        
        result = torch.stack([xy, z_high, z_low], dim=1)
        if self.logger is not None:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        # TODO Add tanh and commands
        H = self.get_foot(q, self.link_name)
        J = self.get_foot_J(q, self.link_name)
        pos = H[:, :3, 3]

        J_xy = 2 * pos[:, 0].unsqueeze(1) * J[:, 0] / (self.alpha ** 2) + 2 * pos[:, 1].unsqueeze(1) * J[:, 1] / (self.beta ** 2)
        J_z_high = J[:, 2]
        J_z_low = -J[:, 2]

        return torch.stack([J_xy, J_z_high, J_z_low], dim=1).to(q.device)
        
def build_atacom_agent(rl_agent, dynamics_info, atacom_params):
    dyn = VelocityControlSystem(dim_q=dynamics_info['n_joints'], vel_limit=dynamics_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=dynamics_info['n_joints'])
    # constr_list.add_constraint(JointPosConstraint(dynamics_info['n_joints'], dynamics_info['joint_pos_limit'].to(TorchUtils.get_device()), logger=dynamics_info['env_info']['logger']))
    
    # No possible action
    constr_list.add_constraint(JointPosConstraint(dynamics_info['n_joints'], torch.vstack([torch.tensor([-2. for _ in range(dynamics_info['n_joints'])], dtype=torch.float32), torch.tensor([2. for _ in range(dynamics_info['n_joints'])], dtype=torch.float32)]).to(TorchUtils.get_device()), logger=dynamics_info['env_info']['logger']))
    
    # constr_list.add_constraint(FootPosConstraint(dynamics_info['n_joints'], 'LF', dynamics_info['env_info']['function']['get_relative_link'], dynamics_info['env_info']['function']['get_J_relative_link'], logger=dynamics_info['env_info']['logger']))
    # constr_list.add_constraint(FootPosConstraint(dynamics_info['n_joints'], 'RF', dynamics_info['env_info']['function']['get_relative_link'], dynamics_info['env_info']['function']['get_J_relative_link'], logger=dynamics_info['env_info']['logger']))
    # constr_list.add_constraint(FootPosConstraint(dynamics_info['n_joints'], 'LH', dynamics_info['env_info']['function']['get_relative_link'], dynamics_info['env_info']['function']['get_J_relative_link'], logger=dynamics_info['env_info']['logger']))
    # constr_list.add_constraint(FootPosConstraint(dynamics_info['n_joints'], 'RH', dynamics_info['env_info']['function']['get_relative_link'], dynamics_info['env_info']['function']['get_J_relative_link'], logger=dynamics_info['env_info']['logger']))
    atacom_controller = ATACOMController(constr_list, dyn,
                                         slack_beta=atacom_params['slack_beta'],
                                         slack_tol=atacom_params['slack_tol'],
                                         slack_dynamics_type=atacom_params['slack_dynamics_type'],
                                         drift_compensation_type=atacom_params['drift_compensation_type'],
                                         drift_clipping=atacom_params['drift_clipping'],
                                         lambda_c=atacom_params['lambda_c'])
    return ATACOMWrapper(env_info=dynamics_info['env_info'],
                         atacom_controller=atacom_controller,
                         learning_agent=rl_agent,
                         randomize_dynamics=atacom_params['randomize_dynamics'])


