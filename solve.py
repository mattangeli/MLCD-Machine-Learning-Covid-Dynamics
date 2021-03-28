from funct import *
import numpy as np
import torch
import matplotlib.pyplot as plt



def solve(model, X0, Xf, nTest):

    t0, x0, lam0 = X0
    tf, xf, lamf = Xf
       
    tTest = torch.linspace(t0,tf,nTest)
    tTest = tTest.reshape(-1,1);
    tTest.requires_grad=True
    
    x0Test = torch.linspace(x0,xf,nTest)
    x0Test = x0Test.reshape(-1,1);
    x0Test.requires_grad=True

    lamTest = torch.linspace(lam0,lamf,nTest)
    lamTest = x0Test.reshape(-1,1);
#    lamTest.requires_grad=True
    t_bundle = torch.cat([tTest,x0Test,lamTest],dim =1)

    dt =tTest-t0
    f = (1-torch.exp(-dt)) 
    nn = model(t_bundle)
#   find solution for each x and lam
    Loss_test_history = []
    x_history = []
    lam_history = []
    for x in x0Test:
         for lam in lamTest:

             x_hat = x + f*nn   
             xdot = dfx(tTest, x_hat)
             ode = xdot + lam*x_hat
             L  = (ode.pow(2)).mean()
             
             Loss_test_history.append(L.detach().numpy())
             x_history.append(x.detach().numpy())
             lam_history.append(lam.detach().numpy())

    

    plt.plot(x_history, Loss_test_history, label='Loss');
    plt.ylabel('x0');plt.xlabel('lam')
    plt.legend()
   
    plt.savefig('losses.png')

    return 

