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
from experiments.util.lie_group import SO3

class ATACOMWrapper(AgentWrapper):
    def __init__(self, env_info, atacom_controller: ATACOMController, learning_agent, randomize_dynamics=False, atacom_enable=None):

        self.env_info = env_info

        super().__init__(atacom_controller=atacom_controller, learning_agent=learning_agent, randomize_dynamics=randomize_dynamics, atacom_enable=atacom_enable)

    def _set_atacom_into_policy(self):
        if hasattr(self.learning_agent.policy, 'set_atacom_controller') and self.atacom_enable:
            self.learning_agent.policy.set_atacom_controller(self.atacom_controller, self.env_info)
        
    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.

    def _void_action(self):
        return torch.zeros((self.env_info['num_envs'], self.env_info['action']['len']), device=TorchUtils.get_device())

class HFEConstraint(Constraint):
    def __init__(self, n_joints, joint_limits, logger=None, check_J=False):
        name = 'HFE_pos'
        self.n_joints = n_joints
        self.logger = logger
        self.joint_limits = joint_limits
        self.check_J = check_J
        super().__init__(name, dim_q=self.n_joints, dim_k=8, dim_z=0)
        
    def fun(self, q, z=None, log=True):
        pos = q[:, :self.n_joints][:, [i*3 + 1 for i in range(4)]]
        result = torch.cat([pos - self.joint_limits[1], self.joint_limits[0] - pos], dim=1)
        if self.logger is not None and log:
            self.logger.log(name=self.name, value=torch.maximum(pos - self.joint_limits[1], self.joint_limits[0] - pos))
        return result.to(q.device)

    def df_dq(self, q, z=None):
        J = torch.zeros(q.shape[0], 4, self.n_joints)
        for i in range(4):
            J[:, i, i*3 + 1] = 1
        J_pos = torch.cat([J, -J], dim=1)
        result = J_pos.to(q.device)
        
        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)
        return result

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
            self.logger.log(name=self.name, value=torch.maximum(pos - self.joint_limits[1], self.joint_limits[0] - pos))
        return result.to(q.device)

    def df_dq(self, q, z=None):
        J_pos = torch.vstack([torch.eye(self.n_joints), -torch.eye(self.n_joints)])
        result = J_pos.unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device)
        
        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)
        return result

    
class MinHeightConstraint(Constraint):
    def __init__(self, side, env_info, dim_k=1, z=-0.2, check_J=False):
        name = side + '_foot_min_height'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.link_name = side + '_foot'
        self.z = z
        self.check_J = check_J
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.foot = LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side])

    def fun(self, q, z=None, log=True):
        pos = self.foot.get_pos(q)

        result = (pos[:, 2] - self.z).unsqueeze(1)

        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result) # value=torch.stack([xy, torch.maximum(z_high, z_low)], dim=1))
        return result.to(q.device)

    def df_dq(self, q, z=None):

        J = self.foot.get_J(q)

        result = J[:, 2].unsqueeze(1)

        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)

        return result
    
class MaxHeightConstraint(Constraint):
    def __init__(self, side, env_info, dim_k=1, z=-0.4, check_J=False):
        name = side + '_foot_max_height'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.link_name = side + '_foot'
        self.z = z
        self.check_J = check_J
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.foot = LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side])

    def fun(self, q, z=None, log=True):
        pos = self.foot.get_pos(q)

        result = (self.z - pos[:, 2]).unsqueeze(1)

        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result) 
        return result.to(q.device)

    def df_dq(self, q, z=None):
        J = self.foot.get_J(q)

        result = -J[:, 2].unsqueeze(1)

        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)

        return result

class FootPosConstraint(Constraint):
    def __init__(self, side, env_info, dim_k=1, alpha=0.3, beta=0.3, use_commands=False, check_J=False):
        name = side + '_foot_pos'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.get_command = env_info['fun']['get_commands']
        self.use_commands = use_commands
        self.alpha = alpha
        self.beta = beta
        self.link_name = side + '_foot'
        self.check_J = check_J
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.foot = LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side])

    def fun(self, q, z=None, log=True):
        pos = self.foot.get_pos(q)
        alpha, beta = self._get_ellipse()

        result = (torch.sqrt(pos[:, 0] ** 2 / (alpha ** 2) + pos[:, 1] ** 2 / (beta ** 2)) - 1).unsqueeze(1)

        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)

    def df_dq(self, q, z=None):
        pos = self.foot.get_pos(q)
        J = self.foot.get_J(q)
        alpha, beta = self._get_ellipse()

        f_sqrt = torch.sqrt(pos[:, 0] ** 2 / (alpha ** 2) + pos[:, 1] ** 2 / (beta ** 2))
        J_pos = torch.stack([pos[:, 0] / (alpha ** 2 * f_sqrt),  pos[:, 1] / (beta ** 2 * f_sqrt)], dim=1).unsqueeze(1)
        result = (J_pos @ J[:, :2]).squeeze(-2).unsqueeze(1).to(q.device)

        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)

        return result
    
    def _get_ellipse(self, min_a=0.2, max_a=1.):
        if not self.use_commands:
            return self.alpha, self.beta
        
        scale_fun = lambda com, val, min, max: torch.minimum(torch.tensor(max * val, device=com.device), torch.maximum(torch.tensor(min * val, device=com.device), val * ((max - min) * com + min)))
        
        commands = self.get_command().abs()
        alpha = scale_fun(commands[:, 0], self.alpha, min_a, max_a)
        beta = scale_fun(commands[:, 1], self.beta, min_a, max_a)

        return alpha, beta
    
class FootRotConstraint(Constraint):
    def __init__(self, env_info, dim_k=4, base_angle=None, min_angle=torch.pi/2, max_angle=torch.pi, check_J=False):
        name = 'Foot_rot'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.base_angle = repeat_until(base_angle, 4) if base_angle is not None else None
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_ground_distance = 0.2
        self.check_J = check_J
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.feet = [LinkPos(env_info['urdf_path'], side + '_foot', side + '_thigh', env_info['default_joint_pos'],  env_info['action']['idx'][side]) for side in ['FL', 'FR', 'RL', 'RR']]
        if base_angle is not None:
            for i, foot in enumerate(self.feet):
                foot.set_base_rot(SO3.Exp(torch.tensor(self.base_angle[i], device=foot.base_matrix.device)).to(foot.base_matrix.dtype))
    
    def fun(self, q, z=None, log=True):
        ground_distance, _ = self.get_ground_distance(q)
        constraint_angle = ground_distance * (self.max_angle - self.min_angle) / self.max_ground_distance + self.min_angle
        result = torch.zeros(q.shape[0], self.dim_k).to(q.device) 
        for i, foot in enumerate(self.feet):
            rot = foot.get_rot(q)
            angle = torch.norm(SO3.Log(rot), dim=1)
            result[:, i] = angle - constraint_angle[:, i]

        if self.logger is not None and log:
            self.logger.log(name=self.name, value=result)
        return result.to(q.device)
    
    def df_dq(self, q, z=None):
        J_ground = self.get_ground_distance_df_dq(q)
        J_constraint_angle = J_ground * (self.max_angle - self.min_angle) / self.max_ground_distance
        result = torch.zeros(q.shape[0], self.dim_k, self.dim_q).to(q.device)
        for i, foot in enumerate(self.feet):
            J_rot = foot.get_J(q)[:, 3:]
            J_angle = (SO3.Log(foot.get_rot(q)) / torch.norm(SO3.Log(foot.get_rot(q)), dim=1).unsqueeze(-1)).unsqueeze(1) @ J_rot
            result[:, i] = J_angle.squeeze(1) - J_constraint_angle[:, i]

        if self.check_J:
            assert check_jacobian(self.fun, result, q, z)
        return result.to(q.device)

    def get_ground_distance(self, q):
        feet_z = torch.stack([foot.get_pos(q)[:, 2] for foot in self.feet], dim=-1)
        min_z = feet_z.min(dim=1)
        return feet_z - min_z.values.unsqueeze(-1), min_z.indices
    
    def get_ground_distance_df_dq(self, q):
        J_ground = torch.eye(len(self.feet)).unsqueeze(0).repeat(q.shape[0], 1, 1).to(q.device) 
        J_z =  torch.stack([foot.get_J(q)[:, 2] for i, foot in enumerate(self.feet)], dim=1)
        _, min_z_idx = self.get_ground_distance(q)
        J_ground[range(q.shape[0]), :, min_z_idx] += -1

        return J_ground @ J_z

def check_jacobian(fun, result, q, z):
    J_num = torch.empty_like(result)[0]
    for i in range(J_num.shape[0]):
        f = lambda x: fun(torch.tensor(x).to('cuda').unsqueeze(0), z, log=False)[0, i].cpu().numpy()
        J_i = numerical_diff_function(f, q[0].cpu().numpy(), eps=1e-5)
        J_num[i] = torch.tensor(J_i).to(q.device)
    return torch.allclose(J_num, result[0])

def build_atacom_agent(rl_agent, env_info, atacom_params, constraints_params):
    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = constraint_list(constraints_params, env_info)

    atacom_controller = ATACOMController(constr_list, dyn,
                                         slack_beta=atacom_params['slack_beta'],
                                         slack_tol=atacom_params['slack_tol'],
                                         slack_dynamics_type=atacom_params['slack_dynamics_type'],
                                         drift_compensation_type=atacom_params['drift_compensation_type'],
                                         drift_clipping=atacom_params['drift_clipping'],
                                         lambda_c=atacom_params['lambda_c'],
                                         lambda_c_i=atacom_params['lambda_c_i'],
                                         integral_window=atacom_params['integral_window'])

    return ATACOMWrapper(env_info=env_info,
                         atacom_controller=atacom_controller,
                         learning_agent=rl_agent,
                         randomize_dynamics=atacom_params['randomize_dynamics'],
                         atacom_enable=atacom_params['enable'])

def constraint_list(constraints_params, env_info):
    constr_list = ConstraintList(dim_q=env_info['n_joints'])

    if constraints_params['hfe_limit']:
        limit = repeat_until(constraints_params['hfe_limit'], 4)
        if constraints_params['joint_percentage']:
            env_limit = env_info['joint_pos_limit'][:, [i*3 + 1 for i in range(4)]].clone().to(TorchUtils.get_device())
            invalid_space = (env_limit[1] - env_limit[0]) * (1 - limit) / 2
            joint_limit = torch.vstack([env_limit[0] + invalid_space, env_limit[1] - invalid_space])
        else:
            joint_limit = torch.vstack([-limit, limit]) + env_info['default_joint_pos']
        constr_list.add_constraint(HFEConstraint(env_info['n_joints'], joint_limit, logger=env_info['logger'], check_J=constraints_params['check_J']))

    if constraints_params['joint_limit']:
        limit = repeat_until(constraints_params['joint_limit'], env_info['n_joints'])
        if constraints_params['joint_percentage']:
            env_limit = env_info['joint_pos_limit'].clone().to(TorchUtils.get_device())
            invalid_space = (env_limit[1] - env_limit[0]) * (1 - limit) / 2
            joint_limit = torch.vstack([env_limit[0] + invalid_space, env_limit[1] - invalid_space])
        else:
            joint_limit = torch.vstack([-limit, limit]) + env_info['default_joint_pos']
        constr_list.add_constraint(JointPosConstraint(env_info['n_joints'], joint_limit, logger=env_info['logger'], check_J=constraints_params['check_J']))

    if constraints_params['feet_pos']:
        for side in constraints_params['feet_pos']:
            if isinstance(constraints_params['foot_pos_alpha'], (int, float)) and isinstance(constraints_params['foot_pos_beta'], (int, float)):
                constr_list.add_constraint(FootPosConstraint(side, env_info, 
                                                                check_J=constraints_params['check_J'], 
                                                                alpha=constraints_params['foot_pos_alpha'],
                                                                beta=constraints_params['foot_pos_beta']))
            if isinstance(constraints_params['foot_pos_min_z'], (int, float)):
                constr_list.add_constraint(MaxHeightConstraint(side, env_info, 
                                                                check_J=constraints_params['check_J'], 
                                                                z=constraints_params['foot_pos_min_z']))
            if isinstance(constraints_params['foot_pos_max_z'], (int, float)):
                constr_list.add_constraint(MinHeightConstraint(side, env_info, 
                                                                check_J=constraints_params['check_J'], 
                                                                z=constraints_params['foot_pos_max_z']))
            
    if constraints_params['foot_rot_max'] and constraints_params['foot_rot_min']:
        constr_list.add_constraint(FootRotConstraint(env_info, 
                                                            base_angle=constraints_params['foot_rot_base'],
                                                            min_angle=constraints_params['foot_rot_min'],
                                                            max_angle=constraints_params['foot_rot_max'],
                                                            check_J=constraints_params['check_J']))

    return constr_list


def repeat_until(values, n):
    if n % len(values) != 0:
        raise Exception('The number of joints must be divisible by the number of joint limits')
    repeat_len = n // len(values)
    t_values = torch.tensor(values, dtype=torch.float32).to(TorchUtils.get_device())
    limit = t_values.repeat((repeat_len, *[1 for _ in range(t_values.ndim - 1)]))

    return limit

