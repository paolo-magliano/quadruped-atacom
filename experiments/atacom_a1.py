import torch
import pytorch_kinematics as pk
import time

from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.utils.numerical_gradient import numerical_diff_function

from atacom import VelocityControlSystem
from atacom import Constraint, ConstraintList
from atacom import ATACOMController
from atacom import AgentWrapper

from experiments.kinematics_a1 import LinkPos

class ATACOMWrapper(AgentWrapper):
    def __init__(self, env_info, atacom_controller: ATACOMController, learning_agent, randomize_dynamics=False, old_atacom_controller=None):

        self.env_info = env_info

        super().__init__(atacom_controller=atacom_controller, learning_agent=learning_agent, randomize_dynamics=randomize_dynamics) #, old_atacom_controller=old_atacom_controller)

    def _set_atacom_into_policy(self):
        if hasattr(self.learning_agent.policy, 'set_atacom_controller'):
            self.learning_agent.policy.set_atacom_controller(self.atacom_controller, self.env_info)
        
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.

    def _void_action(self):
        return torch.zeros((self.env_info['num_envs'], self.env_info['action']['len']), device=TorchUtils.get_device())

class JointPosConstraint(Constraint):
    def __init__(self, n_joints, joint_limits, logger=None, check_J=False):
        name = 'Joint_pos'
        self.n_joints = n_joints
        self.joint_limits = joint_limits
        self.logger = logger
        self.check_J = check_J
        super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints * 2, dim_z=0)

    def fun(self, q, z=None, log=True):
        pos = q[:, :self.n_joints]
        result = torch.cat([pos - self.joint_limits[1], self.joint_limits[0] - pos], dim=1)
        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        J_pos = torch.vstack([torch.eye(self.n_joints), -torch.eye(self.n_joints)])
        result = J_pos.unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device)
        
        if self.check_J:
            check_jacobian(self.fun, result, q, z)
        # J_num = torch.empty_like(result)[0]
        # for i in range(self.dim_k):
        #     f = lambda x: self.fun(torch.tensor(x).to('cuda').unsqueeze(0), z, log=False)[0, i].cpu().numpy()
        #     J_i = numerical_diff_function(f, q[0].cpu().numpy())
        #     J_num[i] = torch.tensor(J_i).to(q.device)
        return result

class FootPosConstraint(Constraint):
    def __init__(self, side, env_info, dim_k=3, alpha=0.5, beta=0.5, min_z=-0.55, max_z=-0.15, use_commands=False, check_J=False):
        name = side + '_foot_pos'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.get_foot = env_info['fun']['get_relative_link']
        self.get_foot_J = env_info['fun']['get_J_relative_link']
        self.get_command = env_info['fun']['get_commands']
        self.use_commands = use_commands
        self.alpha = alpha
        self.beta = beta
        self.link_name = side + '_foot'
        self.min_z = min_z
        self.max_z = max_z
        self.check_J = check_J
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.foot = LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side])

    def fun(self, q, z=None, log=True):
        # TODO Add tanh and commands
        # H = self.get_foot(q, self.link_name)
        # pos = H[:, :3, 3]

        pos = self.foot.get_pos(q)
        alpha, beta = self._get_ellipse()

        xy = pos[:, 0] ** 2 / (alpha ** 2) + pos[:, 1] ** 2 / (beta ** 2) - 1
        z_high = pos[:, 2] - self.max_z
        z_low = self.min_z - pos[:, 2]
        # print(f'Z max: {pos[:, 2].max()} Z min: {pos[:, 2].min()}')

        result = torch.stack([xy, z_high, z_low], dim=1)
        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        # TODO Add tanh and commands
        # H = self.get_foot(q, self.link_name)
        # J = self.get_foot_J(q, self.link_name)
        # pos = H[:, :3, 3]

        pos = self.foot.get_pos(q)
        J = self.foot.get_J(q)
        alpha, beta = self._get_ellipse()

        J_pos = torch.stack([2 * pos[:, 0] / (alpha ** 2), 2 * pos[:, 1] / (beta ** 2)], dim=1).unsqueeze(1)
        J_xy = (J_pos @ J[:, :2]).squeeze(-2)
        J_z_high = J[:, 2]
        J_z_low = -J[:, 2]

        result = torch.stack([J_xy, J_z_high, J_z_low], dim=1).to(q.device)

        if self.check_J:
            check_jacobian(self.fun, result, q, z)

        return result
    
    def _get_ellipse(self, min_a=0.2, max_a=1.):
        if not self.use_commands:
            return self.alpha, self.beta
        
        scale_fun = lambda com, val, min, max: torch.minimum(torch.tensor(max * val, device=com.device), torch.maximum(torch.tensor(min * val, device=com.device), val * ((max - min) * com + min)))
        
        commands = self.get_command().abs()
        alpha = scale_fun(commands[:, 0], self.alpha, min_a, max_a)
        beta = scale_fun(commands[:, 1], self.beta, min_a, max_a)

        return alpha, beta
    
def check_jacobian(fun, result, q, z):
    J_num = torch.empty_like(result)[0]
    for i in range(J_num.shape[0]):
        f = lambda x: fun(torch.tensor(x).to('cuda').unsqueeze(0), z, log=False)[0, i].cpu().numpy()
        J_i = numerical_diff_function(f, q[0].cpu().numpy(), eps=1e-5)
        J_num[i] = torch.tensor(J_i).to(q.device)
    torch.allclose(J_num, result[0])

def build_atacom_agent(rl_agent, env_info, atacom_params, constraints_params):
    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=env_info['n_joints'])

    if constraints_params['joint_limit']:
        if env_info['n_joints'] % len(constraints_params['joint_limit']) != 0:
            raise Exception('The number of joints must be divisible by the number of joint limits')
        repeat_len = env_info['n_joints'] // len(constraints_params['joint_limit'])
        limit = torch.tensor(constraints_params['joint_limit'], dtype=torch.float32).to(TorchUtils.get_device()).repeat(repeat_len)
        joint_limit = torch.vstack([-limit, limit]) + env_info['default_joint_pos']
        constr_list.add_constraint(JointPosConstraint(env_info['n_joints'], joint_limit, logger=env_info['logger'], check_J=constraints_params['check_J']))

    if constraints_params['feet']:
        for side in constraints_params['feet']:
            constr_list.add_constraint(FootPosConstraint(side, env_info, check_J=constraints_params['check_J']))

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

