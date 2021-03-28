import numpy as np
import torch
from torch.autograd import grad

# Calculate the derivatice with auto-differention
def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]

# parametric solutions
def parametricSolutions(t_bundle, nn, X0):
    t0  = X0[0]
    N1  = nn(t_bundle)
    t = t_bundle[:, 0].reshape(-1,1)
    x0s = t_bundle[:, 1].reshape(-1,1)
    dt =t-t0
    f = (1-torch.exp(-dt))
    x_hat  = x0s  + f * (N1 - x0s)
    return x_hat

def Eqs_Loss(t, x_hat, t_bundle):
    xdot = dfx(t, x_hat)
    lam = t_bundle[:, 2].reshape(-1,1)
    ode  = xdot + lam*x_hat
    L  = (ode.pow(2)).mean()
    return L

