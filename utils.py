import numpy as np
import torch

#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
def perturbPoints(v0, vf, n_train, sig=0.5):
    
    grid = torch.linspace(v0, vf, n_train).reshape(-1, 1)
    delta_v = grid[1] - grid[0]  
    noise = delta_v * torch.randn_like(grid)*sig
    v = grid + noise
    v.data[2] = torch.ones(1, 1)*(-1)
    v.data[v<v0]=v0 - v.data[v<v0]
    v.data[v>vf]=2*vf - v.data[v>vf]
    v.requires_grad = True
    
    return v

    

