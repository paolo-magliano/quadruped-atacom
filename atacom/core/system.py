import torch
from abc import ABC, abstractmethod


class ControlAffineSystem(ABC):
    def __init__(self, dim_q, dim_u):
        self.dim_q = dim_q
        self.dim_u = dim_u
        self.u_limit = torch.ones(self.dim_u)

    @abstractmethod
    def f(self, q):
        pass

    @abstractmethod
    def G(self, q):
        pass

    def dq(self, q, u):
        assert u.shape[-1] == self.dim_u
        return self.f(q) + self.G(q) @ u


class VelocityControlSystem(ControlAffineSystem):
    """
    q_dot = vel_limit * u; f(q) = 0; G(q) = vel_limit
    """
    def __init__(self, dim_q, vel_limit):
        super().__init__(dim_q, dim_q)
        assert self.u_limit.shape == vel_limit.shape
        self.u_limit = vel_limit

    def f(self, q):
        assert q.shape[-1] == self.dim_q
        return torch.zeros(self.dim_q, device=q.device)

    def G(self, q):
        assert q.shape[-1] == self.dim_q
        return torch.diag(self.u_limit).to(q.device)


class AccelerationControlSystem(ControlAffineSystem):
    """
    q_ddot = acc_limit * u; f(q) = 0; G(q) = acc_limit
    """
    def __init__(self, dim_q, acc_limit):
        super().__init__(dim_q, dim_q)
        assert self.u_limit.shape == acc_limit.shape
        self.u_limit = acc_limit

    def f(self, q):
        assert q.shape[-1] == self.dim_q
        return torch.zeros(self.dim_q, device=q.device)

    def G(self, q):
        assert q.shape[-1] == self.dim_q
        return torch.diag(self.u_limit).to(q.device)
