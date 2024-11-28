import torch

def qr_null(A, tol=None):
    Q, R = torch.linalg.qr(A.transpose(-1, -2), mode='complete')
    tol = torch.amax(A, dim=(-2, -1)) * torch.finfo(R.dtype).eps if tol is None else tol
    while tol.dim() < R.dim() - 1:
        tol = tol.unsqueeze(-1)
    rnk = torch.full(A.shape[:-2], min(A.shape[-2:])).to(device=A.device) - torch.searchsorted(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1)).flip(dims=(-1,)), tol)
    return Q[..., rnk.min():].conj()


def smooth_basis(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = qr_null(A)
    if T0 is None:
        T0 = torch.zeros(Ux.shape[-2:], device=Ux.device)
        T0.fill_diagonal_(1.0)

    else:
        assert T0.shape == (Ux.shape[-2], Ux.shape[-1])

    U0 = Ux.transpose(-1, -2) @ T0
    U, s, Vh = torch.linalg.svd(U0)
    Q = U @ Vh
    return Ux @ Q
