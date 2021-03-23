import numpy as np
import torch
from torch.autograd import grad



# Calculate the derivatice with auto-differention
def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]


# parametric solutions
def parametricSolutions(t, nn, X0):
    t0, x0  = X0[0],X0[1]
    N1  = nn(t)
    dt =t-t0
    f = (1-torch.exp(-dt))
    x_hat  = x0  + f*N1
    return x_hat


def Eqs_Loss(t,x1, X0):
    xdot = dfx(t,x1)
    lam = X0[2]
    f1  = xdot + lam*x1
    L  = (f1.pow(2)).mean()
    return L

