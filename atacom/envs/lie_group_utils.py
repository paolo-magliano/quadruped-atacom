import matplotlib.pyplot as plt
import torch


class SO3: 
    def Log(R):
        R = torch.tensor(R) if not isinstance(R, torch.Tensor) else R
        angle = torch.acos((torch.trace(R) - 1) / 2)
        w_hat = angle * (R - R.T) / (2 * torch.sin(angle))
        w = torch.tensor([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

        return w

    def Exp(w):
        w = torch.tensor(w) if not isinstance(w, torch.Tensor) else w
        angle = torch.norm(w)
        w_hat = skew_matrix(w)
        R = torch.eye(3) + torch.sin(angle) * w_hat / angle + (1 - torch.cos(angle)) * w_hat @ w_hat / angle**2

        return R
    
    def Ad(R):
        R = torch.tensor(R) if not isinstance(R, torch.Tensor) else R

        return R

    def Jl(w):
        w = torch.tensor(w) if not isinstance(w, torch.Tensor) else w
        angle = torch.norm(w)
        w_hat = skew_matrix(w)

        Jl = torch.eye(3) + (1 - torch.cos(angle)) * w_hat / angle**2 + (angle - torch.sin(angle)) * w_hat @ w_hat / angle**3

        return Jl

    def Jr(w):
        return SO3.Jl(w).T

    def plot(Rs):
        ax = plt.figure().add_subplot(111, projection='3d')

        O = torch.tensor([0, 0, 0])
        base = torch.eye(3)

        ax.quiver(*O, *base[:, 0], color='r')
        ax.quiver(*O, *base[:, 1], color='g')
        ax.quiver(*O, *base[:, 2], color='b')

        for R in Rs:
            base_R = R @ base

            ax.quiver(*O, *base_R[:, 0], color='r', linestyle='dashed')
            ax.quiver(*O, *base_R[:, 1], color='g', linestyle='dashed')
            ax.quiver(*O, *base_R[:, 2], color='b', linestyle='dashed')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        plt.show()

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
    
    def Jr_xy(X, Y, scale=1e-6):
        Jr = torch.zeros(6, 6, dtype=X.dtype)
        for i in range(6):
            tau = torch.zeros(6, dtype=X.dtype)
            tau[i] = scale
            Jr[:, i] = k.Se3.from_matrix(torch.inverse(X @ Y) @ (X @ k.Se3.exp(tau).matrix() @ Y)).log() / scale

        return Jr
    
    def Jr_yx(Y, X, scale=1e-6):
        # SE3.Ad(SE3.Exp(Y)) @ SE3.Jr(X) @ SE3.Ad(SE3.Exp(-Y)) Possible solution
        Jr = torch.zeros(6, 6)

        return Jr

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
    return torch.tensor([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
