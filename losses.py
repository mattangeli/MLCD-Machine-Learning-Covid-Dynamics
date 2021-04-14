import torch
import numpy as np
from torch.autograd import grad
from utils import SAIVR_derivs

def sir_loss(t, s, a, i, v, r, param_bundle, param_fixed, decay= 0.):

    alpha_1, beta_1, gamma = param_bundle[0][:], param_bundle[1][:],  param_bundle[2][:]

    s_prime = dfx(t, s)
    a_prime = dfx(t, a)
    i_prime = dfx(t, i)
    v_prime = dfx(t, v)
    r_prime = dfx(t, r)

    u = [s, a, i, v, r]
    ds, da, di, dv, dr = SAIVR_derivs(u, t, alpha_1, beta_1, gamma, param_fixed)

    loss_s = s_prime - ds
    loss_a = a_prime - da
    loss_i = i_prime - di
    loss_v = v_prime - dv        
    loss_r = r_prime - dr
   
    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t) 
    loss_a = loss_a * torch.exp(-decay * t) 
    loss_i = loss_i * torch.exp(-decay * t) 
    loss_v = loss_v * torch.exp(-decay * t) 
    loss_r = loss_r * torch.exp(-decay * t) 
    
    loss_s = (loss_s.pow(2)).mean() 
    loss_a = (loss_a.pow(2)).mean() 
    loss_i = (loss_i.pow(2)).mean() 
    loss_v = (loss_v.pow(2)).mean() 
    loss_r = (loss_r.pow(2)).mean() 

    total_loss = loss_s + loss_a + loss_i + loss_v + loss_r

    return total_loss


def data_fitting_loss(t, true_pop, s_hat, a_hat, i_hat, v_hat, r_hat, mode = 'mse', data_type = 'synthetic'):

    if data_type == 'synthetic':
       s_true = true_pop[0]
       a_true = true_pop[1]
       i_true = true_pop[2]
       v_true = true_pop[3]
       r_true = true_pop[4]
       
       if mode == 'mse':
           loss_s = (s_true - s_hat).pow(2)
           loss_a = (a_true - a_hat).pow(2)
           loss_i = (i_true - i_hat).pow(2)
           loss_v = (v_true - v_hat).pow(2)
           loss_r = (r_true - r_hat).pow(2)

           losses = [loss_s, loss_a, loss_i, loss_v, loss_r]
       
       elif mode == 'cross_entropy':
           loss_s = - s_true * torch.log(s_hat + 1e-10)
           loss_a = - a_true * torch.log(a_hat + 1e-10)
           loss_i = - i_true * torch.log(i_hat + 1e-10)
           loss_v = - v_true * torch.log(v_hat + 1e-10)
           loss_r = - r_true * torch.log(r_hat + 1e-10)

           losses = [loss_s, loss_a, loss_i, loss_v, loss_r]
       
       else:
           raise ValueError('Invalid loss mode specification!')
       
    else:
        
       i_true = true_pop[1]

       if mode == 'mse':
           losses = (i_true - i_hat).pow(2)             
           
       elif mode == 'cross_entropy':
           losses = - i_true * torch.log(i_hat + 1e-10)
           
       else:
           raise ValueError('Invalid loss mode specification!')

    return losses


def trivial_loss(infected, asymptomatic, hack_trivial):
    trivial_loss = 0.

    for i in infected:
        trivial_loss += i
    
    for i in asymptomatic:
        trivial_loss += i
    
    trivial_loss = hack_trivial * torch.exp(- (trivial_loss/3.) ** 2)
    return trivial_loss

def dfx(x, f):
    # Calculate the derivative with auto-differentiation    
    return grad([f], [x], torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]


