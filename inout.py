import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm


def checkfolders():
    if not os.path.exists('models'):
       os.makedirs('models')

    if not os.path.exists('data'):
       os.makedirs('data')


def printLoss(loss, runTime):
    np.savetxt('data/loss.txt',loss)
    print('Training time (minutes):', runTime/60)
    print('Training Loss: ',  loss[-1] )
    plt.figure()
    plt.loglog(loss,'-b',alpha=0.975);
    plt.tight_layout()
    plt.ylabel('Loss');plt.xlabel('t')

    plt.savefig('nl_ode_loss.png')
    plt.close()

def printGroundThruth(t_net, x_exact, xTest,  xdot_exact, xdotTest):
    lineW = 4 # Line thickness
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.plot(t_net, x_exact,'-g', label='Ground Truth', linewidth=lineW);
    plt.plot(t_net, xTest,'--b', label='Network',linewidth=lineW, alpha=.5);
    plt.ylabel('x(t)');plt.xlabel('t')
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.plot(t_net, xdot_exact,'-g', label='Ground Truth', linewidth=lineW);
    plt.plot(t_net, xdotTest,'--b', label='Network',linewidth=lineW, alpha=.5);
    plt.ylabel('dx\dt');plt.xlabel('t')
    plt.legend()
    
    plt.savefig('simpleExp.png')
    plt.tight_layout()
    plt.close()

def print_scatter(Losses):

    X, Y, Z = Losses[:, 0], Losses[:, 1], Losses[:, 2]
    area = 20.0
    plt.scatter(X,Y,edgecolors='none',s=area,c=Z,
                norm=LogNorm())
    plt.colorbar()
    plt.savefig('scatter_loss.png')
    plt.tight_layout()
    plt.close()
    
    
