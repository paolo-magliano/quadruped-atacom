import torch


class Slack:
    def __init__(self, dim, beta=1., dynamics_type="exp", tol=1e-6, vel_limit=1):
        self.dim = dim
        self.beta = beta
        self.dynamics_type = dynamics_type
        self.tol = tol
        self.vel_limit = vel_limit

    def mu(self, k):
        return torch.maximum(-k, torch.tensor(self.tol, device=k.device))

    def alpha(self, mu):
        mu = torch.atleast_1d(mu)
        if self.dynamics_type == "exp":
            return torch.diag_embed(torch.clip(self.vel_limit * (torch.exp(self.beta * mu) - 1), self.tol, 1 / self.tol))
        elif self.dynamics_type == 'linear':
            return torch.diag_embed(torch.clip(self.vel_limit * self.beta * mu, self.tol, 1 / self.tol))
        else:
            raise NotImplementedError("Unknown type of slack variable")
