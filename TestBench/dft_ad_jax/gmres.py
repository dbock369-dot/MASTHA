import torch
from torch.autograd.functional import vhp
# from torch.linalg import norm
from functools import partial

from typing import Callable


def _givens_rotation(h):
    """Givens rotation as shown in (https://en.wikipedia.org/wiki/Givens_rotation)"""
    if h[1] == 0:
        c = torch.copysign(torch.ones(1), h[0])
        s = torch.zeros(1)
        r = h[0].abs()
    elif h[0] == 0:
        c = torch.zeros(1)
        s = -torch.copysign(torch.ones(1), h[1])
        r = h[1].abs()
    elif h[0].abs() > h[1].abs():
        t = h[1] / h[0]
        u = torch.copysign((1.0 + t * t).sqrt(), h[0])
        c = 1.0 / u
        s = -c * t
        r = h[0] * u
    else:
        t = h[0] / h[1]
        u = torch.copysign((1.0 + t * t).sqrt(), h[1])
        c = t / u
        s = -1.0 / u
        r = h[1] * u
    return r, c, s


def gmres(linear_operator: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, x0: torch.Tensor, max_iter: int, tol: float):
    """
    tol : float
        Absolute tolerance of the residual (could go for relative tolerance w.r.t. the first residual: terminate if `gamma[k] <= tol*gamma[0]`)
    """
    required_iterations = 0
    dimension = rhs.shape.numel()
    iterations = max(min(max_iter, dimension), 0)
    # v = [None]*max_iter  # Krylov space
    v = torch.zeros([iterations + 1, dimension])  # Krylov space
    h = torch.zeros([iterations + 1, iterations])  # Upper Hessenberg matrix
    c = torch.zeros(iterations + 1)  # cos(theta) of Givens rotation angle
    s = torch.zeros(iterations + 1)  # sin(theta) of Givens rotation angle
    gamma = torch.zeros(iterations + 1)  # error of the solution

    # Skip unnecessary linear operator evaluation if initial guess for solution is zero-vector
    if x0.norm().item() != 0.0:
        r = rhs - linear_operator(x0)
    else:
        r = rhs
    gamma[0] = r.norm().item()
    if gamma[0].abs() <= tol:
        return x0, gamma[0].abs().item(), required_iterations

    # stored in first row (last index accessed the fastest)
    v[0] = r / gamma[0]

    # GMRES iterations
    for k in range(iterations):
        v[k+1] = linear_operator(v[k])
        norm_linear_operator = v[k+1].norm().item()  # maybe unnecessary

        # Modified Gram-Schmidt orthogonalization
        h[:k+1, k] = torch.inner(v[:k+1], v[k+1])
        # equals (v[:k+1] * h[:k+1, k:k+1]).sum(dim=0)
        v[k+1] -= torch.tensordot(v[:k+1], h[:k+1, k], dims=([0], [0]))
        h[k+1, k] = torch.linalg.norm(v[k+1])
        norm_new_vector = h[k+1, k].item()  # maybe unnecessary

        # Alternativ (maybe better?)
        # for j in range(k+1):
        #     h[j, k] = torch.dot(v[j], v[k+1])
        #     v[k+1] -= h[j, k] * v[j]
        # h[k+1, k] = torch.linalg.norm(v[k+1])

        # Reorthogonalization
        if norm_linear_operator + 1.0e-3 * norm_new_vector == norm_linear_operator:
            for j in range(k+1):
                h_temp = torch.dot(v[j], v[k+1])
                h[j, k] += h_temp
                v[k+1] -= h_temp * v[j]
            h[k+1, k] = torch.linalg.norm(v[k+1])

        # Givens rotation
        if k > 0:
            for j in range(k):
                givens = torch.tensor([[c[j+1], -s[j+1]],
                                       [s[j+1],  c[j+1]]])
                h[j:j+2, k] = torch.matmul(givens, h[j:j+2, k])

        h[k, k], c[k+1], s[k+1] = _givens_rotation(h[k:k+2, k])
        # Next 2 lines could be replaced by "_givens_rotation(gamma[k:k+2])"
        gamma[k+1] = s[k+1] * gamma[k]
        gamma[k] *= c[k+1]

        # New Krylov vector or termination
        if k < iterations - 1 and gamma[k+1].abs() >= tol:
            # Catch zero denominator
            if h[k+1, k] != 0.0:
                v[k+1] /= h[k+1, k]
            else:
                required_iterations = k + 1
                break
        else:
            required_iterations = k + 1
            break

    siter = slice(0, required_iterations)
    y = torch.linalg.solve_triangular(
        h[siter, siter], gamma[siter, None], upper=True).squeeze(dim=1)

    # return x0 + (v[:required_iterations] * y[:required_iterations, None]).sum(dim=0)
    return x0 + torch.tensordot(v[siter], y, dims=([0], [0])), gamma[required_iterations].abs().item(), required_iterations
