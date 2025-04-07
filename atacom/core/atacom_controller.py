import torch

from typing import Union, Optional, List, Set, Dict, Tuple
from scipy import linalg, optimize
from .constraints import ConstraintList
from .system import ControlAffineSystem
from .slack import Slack
from .utils import smooth_basis


class ATACOMController:
    def __init__(self, constraints: ConstraintList, system: ControlAffineSystem, eq_constraints=None,
                 slack_beta=3., slack_dynamics_type="exp", drift_compensation_type='vanilla', drift_clipping=True,
                 slack_tol=1e-6, lambda_c=1, slack_vel_limit=1., second_order=False):
        self.constraints = constraints
        self.eq_constraints = eq_constraints
        self.system_dynamics = system
        self.dim_q = self.system_dynamics.dim_q
        self.dim_u = self.system_dynamics.dim_u
        self.dim_z = self.constraints.dim_z
        self.dim_k = self.constraints.dim_k
        self.dim_k_eq = 0
        if self.eq_constraints is not None:
            self.dim_k_eq = self.eq_constraints.dim_k
        self.second_order = second_order
        self.lambda_c = lambda_c
        self.drift_compensation_type = drift_compensation_type
        self.drift_clipping = drift_clipping
        self.slack = Slack(self.constraints.dim_k, beta=slack_beta, dynamics_type=slack_dynamics_type, tol=slack_tol,
                           vel_limit=slack_vel_limit)

    def get_q(self, s):
        """
        The state should be ordered as s = [q q_dot z]. 
        """
        return s[..., :self.dim_q]
    
    def get_q_dot(self, s):
        """
        The state should be ordered as s = [q q_dot z]. When not second_order system, return None
        """
        if self.second_order:
            return s[..., self.dim_q:self.dim_q * 2]
        else:
            return None

    def get_z(self, s):
        """
        The state should be ordered as s = [q q_dot z]. When dim_z == 0, return None
        """
        if self.dim_z != 0:
            return s[..., -self.dim_z:]
        else:
            return None

    def compose_action(self, s, u, z_dot=0.):
        # Get State
        q = self.get_q(s)
        z = self.get_z(s)
        q_dot = self.get_q_dot(s)

        # Get Constraint and Residual
        k = self.constraints.k(q, z)
        if self.second_order:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            assert hasattr(self.constraints, 'zeta')
            k = self.constraints.zeta * k + self.constraints.J_q(q, z) @ q_dot
        mu = self.get_mu(k)
        residual = k + mu
        if self.dim_k_eq > 0:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            l = self.eq_constraints.k(q, z)
            residual = torch.cat([residual, l], axis=-1)

        # Get Drift
        psi = self.psi(q, q_dot, z, z_dot)
        J_G = self.J_G(q, z)
        J_u = self.J_u(J_G, mu)

        constraint_direction = self.constraint_direction(J_G, psi, u)
        tangent_index = (constraint_direction > 0).repeat_interleave(3, dim=1)

        if self.drift_compensation_type == 'vanilla':
            u_drift_compensation = torch.linalg.lstsq(J_u, -psi - self.lambda_c * residual).solution
        elif self.drift_compensation_type == 'enforced':
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            u_drift_compensation = torch.linalg.lstsq(J_G, -psi).solution
            u_drift_compensation = torch.cat([u_drift_compensation, torch.zeros(self.constraints.dim_k, device=u_drift_compensation.device)])
            u_drift_compensation += torch.linalg.lstsq(J_u, -self.lambda_c * residual).solution
        elif self.drift_compensation_type == 'modified':
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            # Check the compensation for each individual constriant
            u_comp_individual = [-torch.linalg.pinv(J_G[i:i + 1]) @ psi[i:i + 1] for i in range(self.constraints.dim_k)]
            scale = torch.clip((2 - 2 * torch.abs(torch.vstack(u_comp_individual)).max(axis=1)), 1e-3, 1)
            eff_scale_idx = scale < 1
            if torch.sum(eff_scale_idx) > self.dim_u:
                # Sort by the closeness to the boundary mu of the effective one
                eff_scale = scale[eff_scale_idx]
                eff_scale[torch.argsort(mu[eff_scale_idx])[self.dim_u:]] = 1
                scale[eff_scale_idx] = eff_scale
            J_u = torch.hstack([J_G, scale * self.slack.alpha(mu)])
            u_drift_compensation = torch.linalg.lstsq(J_u, -psi - self.lambda_c * residual).solution


        B_u = smooth_basis(J_u)[..., :self.system_dynamics.dim_u]
        self.u_auxiliary = u_drift_compensation[..., :-self.constraints.dim_k]
        self.u_tangent = B_u[..., :-self.constraints.dim_k, :] @ u.unsqueeze(-1)
        self.u_tangent = self.u_tangent.squeeze(-1)
        self.u_tangent[~tangent_index] = u[~tangent_index]
        u_s = self.u_auxiliary + self.u_tangent
        return u_s, B_u

    def get_mu(self, k):
        """
        q is controllable state, z is uncontrollable state
        """
        return self.slack.mu(k)

    def G_aug(self, q, mu):
        """
        G_aug = [G(q) 0; 0 A(mu)]
        """
        return linalg.block_diag(self.system_dynamics.G(q), self.slack.alpha(mu))

    def J_c(self, q, z):
        """
        J_c = [J_k(q, z) I; J_l(q, z)]
        """
        J_q = self.constraints.J_q(q, z)
        J_c = torch.hstack([J_q, torch.eye(self.constraints.dim_k, device=J_q.device)])
        if self.dim_k_eq > 0:
            J_q = self.eq_constraints.J_q(q, z)
            N = torch.zeros((self.eq_constraints.dim_k, self.constraints.dim_k), device=J_q.device)
            J_c = torch.vstack([J_q, N])
        return J_c

    def J_G(self, q, z):
        J_q = self.constraints.J_q(q, z)
        if self.dim_k_eq > 0:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            J_q = torch.vstack([J_q, self.eq_constraints.J_q(q, z)])

        G = self.system_dynamics.G(q)
        return J_q @ G

    def J_u(self, J_G, mu):
        """
        J_u = J_c @ G_aug = [J_k(q, z)G(q) A(mu); J_l(q, z)G(q) 0]
        """
        A_mu = self.slack.alpha(mu)
        if self.dim_k_eq > 0:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            N = torch.zeros((self.eq_constraints.dim_k, self.constraints.dim_k), device=A_mu.device)
            A_mu = torch.vstack([A_mu, N])
        return torch.cat([J_G, A_mu], dim=-1)

    def psi(self, q, q_dot: Optional[List[float]], z: Optional[List[float]], z_dot: Optional[List[float]]):
        """
        psi = J_q(q, z)f(q) + J_zeta J_z(q, z)z_dot + J_zeta J_q(q, z)q_dot + J_q_dot(q, z, q_dot) q_dot
        """
        f = self.system_dynamics.f(q)
        psi = self.constraints.J_q(q, z) @ f
        if self.dim_z != 0:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            psi += self.constraints.J_z(q, z) @ z_dot

        if self.second_order:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            psi += (self.constraints.zeta[:, None] * self.constraints.J_q(q, z) + self.constraints.J_q_dot(q, q_dot, z)) @ q_dot
            # psi += (self.constraints.zeta[:, None] * self.constraints.J_q(q, z)) @ q_dot

        if self.drift_clipping:
            psi = torch.maximum(psi, torch.tensor(0., device=psi.device))

        if self.dim_k_eq > 0:
            #TODO Check if it works with parallel environment (len(q.shape) > 1)
            psi_eq = self.eq_constraints.J_q(q, z) @ f
            if self.dim_z != 0:
                psi_eq += self.eq_constraints.J_z(q, z) @ z_dot

            psi = torch.cat([psi, psi_eq], axis=-1)
        return psi

    def constraint_direction(self, J_G, psi, u):
        return psi + (J_G @ u.unsqueeze(-1)).squeeze(-1)