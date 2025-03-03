import torch
import pytorch_kinematics as pk
import time

from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.utils.numerical_gradient import numerical_diff_function

from atacom import VelocityControlSystem
from atacom import Constraint, ConstraintList
from atacom import ATACOMController
from atacom import AgentWrapper

from atacom_a1_old import build_old_atacom_controller
from atacom.core.atacom_controller_debug import ATACOMController as ATACOMControllerDebug
from atacom.core.rl_wrapper_debug import AgentWrapper as AgentWrapperDebug

class ATACOMWrapper(AgentWrapper):
    def __init__(self, env_info, atacom_controller: ATACOMController, learning_agent, randomize_dynamics=False, old_atacom_controller=None):

        self.env_info = env_info

        super().__init__(atacom_controller=atacom_controller, learning_agent=learning_agent, randomize_dynamics=randomize_dynamics) #, old_atacom_controller=old_atacom_controller)

    def _unwrap_state(self, obs):
        return obs[:, self.env_info['obs']['joint_pos_idx']] + self.env_info['default_joint_pos'], 0.

    def _void_action(self):
        return torch.zeros((self.env_info['num_envs'], self.env_info['action']['len']), device=TorchUtils.get_device())

class JointPosConstraint(Constraint):
    def __init__(self, n_joints, joint_limits, logger=None):
        name = 'Joint_pos'
        self.n_joints = n_joints
        self.joint_limits = joint_limits
        self.logger = logger
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
        # J_num = torch.empty_like(result)[0]
        # for i in range(self.dim_k):
        #     f = lambda x: self.fun(torch.tensor(x).to('cuda').unsqueeze(0), z, log=False)[0, i].cpu().numpy()
        #     J_i = numerical_diff_function(f, q[0].cpu().numpy())
        #     J_num[i] = torch.tensor(J_i).to(q.device)
        return result

class FootPosConstraint(Constraint):
    def __init__(self, side, env_info, alpha=0.01, beta=0.01, min_z=-0.4, max_z=0, dim_k=3):
        name = side + '_foot_pos'
        self.logger = env_info['logger'] if 'logger' in env_info else None
        self.get_foot = env_info['fun']['get_relative_link']
        self.get_foot_J = env_info['fun']['get_J_relative_link']
        self.get_command = env_info['fun']['get_commands']
        self.alpha = alpha
        self.beta = beta
        self.link_name = side + '_foot'
        self.min_z = min_z
        self.max_z = max_z
        super().__init__(name, dim_q=env_info['n_joints'], dim_k=dim_k, dim_z=0)

        self.q_idx = env_info['action']['idx'][side]
        self.chain = pk.build_chain_from_urdf(open(env_info['urdf_path'], mode="rb").read())
        hip_chain = pk.SerialChain(self.chain, side + '_thigh').to(dtype=torch.float32, device=TorchUtils.get_device())
        self.thigh_pos = hip_chain.forward_kinematics(env_info['default_joint_pos'][self.q_idx][:2]).get_matrix()[0, :3, 3]
        self.foot_chain = pk.SerialChain(self.chain, side + '_foot').to(dtype=torch.float32, device=TorchUtils.get_device())

    def fun(self, q, z=None, log=True):
        # TODO Add tanh and commands
        # H = self.get_foot(q, self.link_name)
        # pos = H[:, :3, 3]

        pos = self.foot_chain.forward_kinematics(q[:, self.q_idx]).get_matrix()[:, :3, 3] - self.thigh_pos

        com = self.get_command()

        # xy = pos[:, 0] ** 2 / (com[:, 0] * self.alpha ** 2) + pos[:, 1] ** 2 / (com[:, 1] * self.beta ** 2) - 1
        xy = pos[:, 0] ** 2 / ( self.alpha ** 2) + pos[:, 1] ** 2 / ( self.beta ** 2) - 1
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

        pos = self.foot_chain.forward_kinematics(q[:, self.q_idx]).get_matrix()[:, :3, 3] - self.thigh_pos
        J = torch.zeros((q.shape[0], 6, q.shape[-1])).to(q.device)
        J[:, :, self.q_idx] = self.foot_chain.jacobian(q[:, self.q_idx])

        J_pos = torch.stack([2 * pos[:, 0] / (self.alpha ** 2), 2 * pos[:, 1] / (self.beta ** 2)], dim=1).unsqueeze(1)
        J_xy = (J_pos @ J[:, :2]).squeeze(-2)
        J_z_high = J[:, 2]
        J_z_low = -J[:, 2]

        result = torch.stack([J_xy, J_z_high, J_z_low], dim=1).to(q.device)

        # J_num = torch.empty_like(result)[0]
        # for i in range(self.dim_k):
        #     f = lambda x: self.fun(torch.tensor(x).to('cuda').unsqueeze(0), z, log=False)[0, i].cpu().numpy()
        #     J_i = numerical_diff_function(f, q[0].cpu().numpy(), eps=1e-5)
        #     J_num[i] = torch.tensor(J_i).to(q.device)
        # torch.allclose(J_num[0], J_xy[0])
        # torch.allclose(J_num[1], J_z_high[0])
        # torch.allclose(J_num[2], J_z_low[0])
        return result

def build_atacom_agent(rl_agent, env_info, atacom_params):
    dyn = VelocityControlSystem(dim_q=env_info['n_joints'], vel_limit=env_info['joint_vel_limit'][1])
    constr_list = ConstraintList(dim_q=env_info['n_joints'])

    up_limit = torch.tensor([0.85, 0.6, 0.6], dtype=torch.float32).to(TorchUtils.get_device()).repeat(4) + env_info['default_joint_pos']
    low_limit = torch.tensor([-0.85, -0.8, -0.7], dtype=torch.float32).to(TorchUtils.get_device()).repeat(4) + env_info['default_joint_pos']
    joint_limit = torch.vstack([low_limit, up_limit])
    limit = torch.tensor([1.5 for _ in range(env_info['n_joints'])], dtype=torch.float32).to(TorchUtils.get_device())
    # joint_limit = torch.vstack([-limit, limit]) + env_info['default_joint_pos']
    constr_list.add_constraint(JointPosConstraint(env_info['n_joints'], joint_limit, logger=env_info['logger']))

    # constr_list.add_constraint(FootPosConstraint('FL', env_info))
    # constr_list.add_constraint(FootPosConstraint('FR', env_info))
    # constr_list.add_constraint(FootPosConstraint('RL', env_info))
    # constr_list.add_constraint(FootPosConstraint('RR', env_info))
    atacom_controller = ATACOMController(constr_list, dyn,
                                         slack_beta=atacom_params['slack_beta'],
                                         slack_tol=atacom_params['slack_tol'],
                                         slack_dynamics_type=atacom_params['slack_dynamics_type'],
                                         drift_compensation_type=atacom_params['drift_compensation_type'],
                                         drift_clipping=atacom_params['drift_clipping'],
                                         lambda_c=atacom_params['lambda_c'])

    old_atacom_controller = build_old_atacom_controller(env_info, atacom_params)
    return ATACOMWrapper(env_info=env_info,
                         atacom_controller=atacom_controller,
                         learning_agent=rl_agent,
                         randomize_dynamics=atacom_params['randomize_dynamics'],
                         old_atacom_controller=old_atacom_controller)

