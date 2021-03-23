#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
from training import run_odeNet
from inout import *
from test import *

checkfolders()

# Set the time range and the training points N
# t0, t_max, N = 0.,  .5*np.pi, 500;  

t0, t_max, N = 0.,  2, 500;     
# Set the initial state. lam controls the nonlinearity
x0 = 1

lam=.5
X0 = [t0, x0,lam]


n_train, neurons, epochs = N, 80, int(3e4)
minibatch_number, minLoss, lr  = 1, 1e-6, 3e-3

model, loss, runTime = run_odeNet(X0, t_max, neurons, epochs, n_train, lr,
        minibatch_number, minLoss, loadWeights = False, PATH = 'models/expDE')

printLoss(loss, runTime)



# compare the predictions with the groud truth

nTest = 10*N ; t_max_test = 1.0*t_max ; t0 = 0.

test_net, x_exact, xTest,  xdot_exact, xdotTest = test(model, X0, nTest, t_max_test)
printGroundThruth(test_net, x_exact, xTest,  xdot_exact, xdotTest)


