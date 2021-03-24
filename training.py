import numpy as np
import torch
import torch.optim as optim
import time
import copy
import os
import sys
from network import odeNet
from utils import perturbPoints
from funct import *
from os import path
import matplotlib.pyplot as plt



# Train the NN
def run_odeNet(X0, Xf,layers, hidden_units, activation, epochs, n_train,lr, betas,
                    minibatch_number, minLoss, loadWeights=False, PATH= "models/expDE"):

    fc0 = odeNet(layers, hidden_units, activation)
    fc1 =  copy.deepcopy(fc0) # fc1 is a deepcopy of the network with the lowest training loss

    optimizer = optim.Adam(fc0.parameters(), lr, betas)
    Loss_history = [];     Llim =  1 
        
    t0=X0[0];
    tf=Xf[0]   

## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval
    
    
## TRAINING ITERATION    
    TeP0 = time.time()
    for epoch in range(epochs):   

# Perturbing the evaluation points & forcing t[0]=t0
        t=perturbPoints(t0, tf, n_train, sig= 0.3*tf)
            
#  Network solution and loss
        x = parametricSolutions(t,fc0,X0)
        Ltot = Eqs_Loss(t,x, X0)            

# OPTIMIZER
        Ltot.backward(retain_graph=False); 
        optimizer.step();   optimizer.zero_grad()

        Loss_history.append(Ltot.detach().numpy())

#Keep the best model (lowest loss) by using a deep copy
        if  epoch > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot 

# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print('Reach minimum requested loss')
            break

    TePf = time.time()
    runTime = TePf - TeP0        
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': Ltot,
    }, PATH)
    
    return fc1, Loss_history, runTime



def loadModel(PATH):
    if path.exists(PATH):
        fc0 = odeNet(layers, hidden_units, activation)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0


   


