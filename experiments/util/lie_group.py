import matplotlib.pyplot as plt
import torch

class SO3:
    def vee(w_hat):
        return torch.stack([w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], dim=-1)
    
    def hat(w):
        return skew_matrix(w)
    
    def Log(R):
        angle = torch.acos((torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).unsqueeze(-1).unsqueeze(-1)
        w_hat = angle * (R - R.mT) / (2 * torch.sin(angle))
        w = SO3.vee(w_hat)

        return w

    def Exp(w):
        angle = torch.norm(w, dim=-1).unsqueeze(-1).unsqueeze(-1)
        w_hat = SO3.hat(w)
        R  = torch.eye(3).unsqueeze(0).repeat(w.shape[0], 1, 1).to(w.device) + torch.sin(angle) * w_hat / angle + (1 - torch.cos(angle)) * w_hat @ w_hat / torch.pow(angle, 2)

        return R
    
    def Ad(R):
        return R

    def Jl(w):
        angle = torch.norm(w, dim=-1).unsqueeze(-1).unsqueeze(-1)
        w_hat = SO3.hat(w)

        Jl = torch.eye(3).unsqueeze(0).repeat(w.shape[0], 1, 1).to(w.device) + (1 - torch.cos(angle)) * w_hat / torch.pow(angle, 2) + (angle - torch.sin(angle)) * w_hat @ w_hat / torch.pow(angle, 3)

        return Jl

    def Jr(w):
        return SO3.Jl(w).mT

class SE3:
    def Log(H):
        H = torch.tensor(H) if not isinstance(H, torch.Tensor) else H
        R = H[:3, :3]
        d = H[:3, 3]

        angle = torch.acos((torch.trace(R) - 1) / 2)
        w_hat = angle * (R - R.T) / (2 * torch.sin(angle))
        w = torch.tensor([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

        V = torch.eye(3) + (1 - torch.cos(angle)) * w_hat / angle**2 + (angle - torch.sin(angle)) * w_hat @ w_hat / angle**3
        p = torch.linalg.inv(V) @ d

        tau = torch.cat((p, w))

        return tau

    def Exp(tau):
        tau = torch.tensor(tau) if not isinstance(tau, torch.Tensor) else tau
        p = tau[:3]
        w = tau[3:]

        angle = torch.norm(w)
        w_hat = skew_matrix(w)
        R = torch.eye(3) + torch.sin(angle) * w_hat / angle + (1 - torch.cos(angle)) * w_hat @ w_hat / angle**2

        V = torch.eye(3) + (1 - torch.cos(angle)) * w_hat / angle**2 + (angle - torch.sin(angle)) * w_hat @ w_hat / angle**3
        d = V @ p

        H = torch.cat((torch.cat((R, d.unsqueeze(1)), dim=1), torch.tensor([[0, 0, 0, 1]])))

        return H
    
    def Ad(H):
        H = torch.tensor(H) if not isinstance(H, torch.Tensor) else H
        R = H[:3, :3]
        t = H[:3, 3]
        t_R = skew_matrix(t) @  R
        Ad = torch.cat([torch.cat([R, t_R], dim=1), torch.cat([torch.zeros(3, 3), R], dim=1)], dim=0)

        return Ad

    def Jl(tau):
        tau = torch.tensor(tau) if not isinstance(tau, torch.Tensor) else tau
        p = tau[:3]
        w = tau[3:]

        angle = torch.norm(w)
        w_hat = skew_matrix(w)
        p_hat = skew_matrix(p)

        q = p_hat / 2 + (angle - torch.sin(angle)) * (w_hat @ p_hat + p_hat @ w_hat + w_hat @ p_hat @ w_hat) / angle**3 - (1 - angle**2 / 2 - torch.cos(angle)) * (w_hat @ w_hat @ p_hat + p_hat @ w_hat @ w_hat - 3 * w_hat @ p_hat @ w_hat) / angle**4 - ((1 - angle**2 / 2 - torch.cos(angle)) / angle**4 - 3 * (angle - torch.sin(angle) - angle**3 / 6) / angle**5) * (w_hat @ p_hat @ w_hat @ w_hat + w_hat @ w_hat @ p_hat @ w_hat) / 2

        Jl = torch.cat((torch.cat((SO3.Jl(w), q), dim=1), torch.cat((torch.zeros((3, 3)), SO3.Jl(w)), dim=1)))

        return Jl

    def Jr(tau):
        return SE3.Jl(-tau)

    def plot(Hs):
        ax = plt.figure().add_subplot(111, projection='3d')

        O = torch.tensor([0, 0, 0])
        base = torch.eye(3)

        ax.quiver(*O, *base[:, 0], color='r')
        ax.quiver(*O, *base[:, 1], color='g')
        ax.quiver(*O, *base[:, 2], color='b')

        x_min, x_max = 0, 0
        y_min, y_max = 0, 0
        z_min, z_max = 0, 0

        for H in Hs:
            O_H = (H @ torch.tensor([*O, 1], dtype=torch.float))[:-1]
            base_H = torch.stack([(H @ torch.tensor([*base[i], 1], dtype=torch.float))[:-1] for i in range(3)])

            ax.quiver(*O, *O_H, color='k', linestyle='dotted')
            ax.quiver(*O_H, *(base_H[:, 0] - O_H), color='r', linestyle='dashed') 
            ax.quiver(*O_H, *(base_H[:, 1] - O_H), color='g', linestyle='dashed')
            ax.quiver(*O_H, *(base_H[:, 2] - O_H), color='b', linestyle='dashed')

            x_min, x_max = min(x_min, O_H[0]), max(x_max, O_H[0])
            y_min, y_max = min(y_min, O_H[1]), max(y_max, O_H[1])
            z_min, z_max = min(z_min, O_H[2]), max(z_max, O_H[2])

        ax.set_xlim([x_min - 1, x_max + 1])
        ax.set_ylim([y_min - 1, y_max + 1])
        ax.set_zlim([z_min - 1, z_max + 1])

        plt.show()
    
def skew_matrix(v):
    zeros = torch.zeros(v.shape[0]).to(v.device)
    row_0 = torch.stack([zeros, -v[:, 2], v[:, 1]], dim=1)
    row_1 = torch.stack([v[:, 2], zeros, -v[:, 0]], dim=1)
    row_2 = torch.stack([-v[:, 1], v[:, 0], zeros], dim=1)
    return torch.stack((row_0, row_1, row_2), dim=1)



