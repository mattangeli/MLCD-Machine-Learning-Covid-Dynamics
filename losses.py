import torch
import numpy as np
from torch.autograd import grad


def sir_loss(t, s, i, r, param_bundle, decay= 0.):
    beta, gamma = param_bundle[0][:], param_bundle[1][:]
    
    s_prime = dfx(t, s)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)

    N = 1

    loss_s = s_prime + (beta * i * s) / N
    loss_i = i_prime - (beta * i * s) / N + gamma * i
    loss_r = r_prime - gamma * i
    
    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)
    
    loss_s = (loss_s.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()

    total_loss = loss_s + loss_i + loss_r

    return total_loss


def trivial_loss(infected, hack_trivial):
    trivial_loss = 0.

    for i in infected:
        trivial_loss += i

    trivial_loss = hack_trivial * torch.exp(- (trivial_loss) ** 2)
    return trivial_loss

def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    
    return grad([f], [x], torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]


