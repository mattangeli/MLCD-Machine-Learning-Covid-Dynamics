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


def data_fitting_loss(t, true_pop, s_hat, i_hat, r_hat, mode = 'mse'):

    s_true = true_pop[0]
    i_true = true_pop[1]
    r_true = true_pop[2]

    if mode == 'mse':
        loss_s = (s_true - s_hat).pow(2)
        loss_i = (i_true - i_hat).pow(2)
        loss_r = (r_true - r_hat).pow(2)
    elif mode == 'cross_entropy':
        loss_s = - s_true * torch.log(s_hat + 1e-10)
        loss_i = - i_true * torch.log(i_hat + 1e-10)
        loss_r = - r_true * torch.log(r_hat + 1e-10)
    else:
        raise ValueError('Invalid loss mode specification!')
 
    return loss_s, loss_i, loss_r



def trivial_loss(infected, hack_trivial):
    trivial_loss = 0.

    for i in infected:
        trivial_loss += i

    trivial_loss = hack_trivial * torch.exp(- (trivial_loss) ** 2)
    return trivial_loss

def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    
    return grad([f], [x], torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]


