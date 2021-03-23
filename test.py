from funct import *
import numpy as np
import torch



def test(model, X0, nTest, t_max_test):

    t0, x0, lam = X0
    tTest = torch.linspace(t0,t_max_test,nTest)

    tTest = tTest.reshape(-1,1);
    tTest.requires_grad=True

    t_net = tTest.detach().numpy()
    
    xTest=parametricSolutions(tTest,model,X0)
    xdotTest=dfx(tTest,xTest)

    xTest=xTest.data.numpy()
    xdotTest=xdotTest.data.numpy()
    
    x_exact = X0[1]*np.exp(-lam*t_net)
    xdot_exact = -lam*x0*np.exp(-lam*t_net)

    return t_net, x_exact, xTest,  xdot_exact, xdotTest

