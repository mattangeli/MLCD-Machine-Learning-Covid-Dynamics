from funct import *
import numpy as np
import torch
from numpy.random import uniform


def test_ode_solution(model, t0, tf, in_test, nTest):

    x0, lam = in_test

    tTest = torch.linspace(t0,tf,nTest).reshape((-1, 1))
    tTest.requires_grad=True
    x0s = x0 * torch.ones(nTest).reshape((-1, 1))
    lams = lam * torch.ones(nTest).reshape((-1, 1))
    t_bundle = torch.cat([tTest,x0s,lams],dim =1)
    t_net = tTest.detach().numpy()
    
    xTest=parametricSolutions(t_bundle,model, [t0, x0, lam])
    xdotTest=dfx(tTest,xTest)

    ode  = xdotTest + lam * xTest
    L  = (ode.pow(2)).mean()

    xTest=xTest.data.numpy()
    xdotTest=xdotTest.data.numpy()
    
    x_exact = x0 * np.exp(-lam * t_net)
    xdot_exact = -lam * x0 * np.exp(-lam * t_net)

    return t_net, x_exact, xTest,  xdot_exact, xdotTest
    
    
def   test_ode_solution_bundle(model, X0, Xf, ntTest, nxTest):

    t0, x0, lam0 = X0
    tf, x0f, lamf = Xf
    
    x0set = torch.linspace(x0,x0f,nxTest)
    lamset = torch.linspace(lam0,lamf,nxTest)
    
    xx, ll, lol = [], [], [] 
      
    for x0 in x0set:
        for lam in lamset:
    
            tTest = torch.linspace(t0,tf,ntTest).reshape((-1, 1))
            tTest.requires_grad=True
            x0s = x0 * torch.ones(ntTest).reshape((-1, 1))
            lams = lam * torch.ones(ntTest).reshape((-1, 1))
            t_bundle = torch.cat([tTest,x0s,lams],dim =1)
            t_net = tTest.detach().numpy()
    
            xTest=parametricSolutions(t_bundle, model, [t0, x0, lam])
            xdotTest=dfx(tTest,xTest)

            ode  = xdotTest + lam * xTest
            L  = (ode.pow(2)).mean()
            
            xx.append(x0.item())
            ll.append(lam.item())
            lol.append(L.item())
              
    Losses = np.column_stack((xx, ll, lol))

    return Losses
    
