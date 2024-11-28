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
    def __init__(self, n_joints, joint_limits):
        name = 'joint_pos'
        self.n_joints = n_joints
        self.joint_limits = joint_limits
        super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints * 2, dim_z=0)

    def fun(self, q, z=None):
        pos = q[:, :self.n_joints]
        result = torch.cat([pos - self.joint_limits[1], self.joint_limits[0] - pos], dim=1)
        return result

    def df_dq(self, q, z=None):
        J_pos = torch.vstack([torch.eye(self.n_joints), -torch.eye(self.n_joints)])
        return J_pos.unsqueeze(0).repeat(q.shape[0], 1, 1)
        
def build_atacom_agent(rl_agent, dynamics_info, atacom_params):
    dyn = VelocityControlSystem(dim_q=dynamics_info['n_joints'], vel_limit=dynamics_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=dynamics_info['n_joints'])
    constr_list.add_constraint(JointPosConstraint(dynamics_info['n_joints'], dynamics_info['joint_pos_limit'].to(TorchUtils.get_device())))
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


