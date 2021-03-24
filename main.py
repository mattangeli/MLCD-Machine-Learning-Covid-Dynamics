#!/usr/bin/env python
# coding: utf-8
from training import run_odeNet
from inout import *
from test import *


if __name__ == '__main__':

   checkfolders()
   
   # define the bundles
   t0, tf = 0.,  2
   x0, xf = 1., 2.
   lam0, lamf = 0.4, 0.8

   X0 = [t0,x0,lam0]
   Xf = [tf, xf, lamf]
   
   # neural network and optimizer parameters
   layers, hidden_units, activation = 2, 30, 'Tanh'
   n_train, neurons, epochs = 500, 80, int(3e4)
   minibatch_number, minLoss, lr  = 1, 1e-6, 3e-3
   betas = [0.9, 0.999]    
   
   # train the model
   model, loss, runTime = run_odeNet(X0, Xf, layers, hidden_units, activation, epochs,
      n_train, lr, betas, minibatch_number, minLoss, loadWeights = False, PATH = 'models/expDE')
   
   printLoss(loss, runTime)
   
   
   # compare the predictions with the groud truth
   nTest = 10*n_train ; t_max_test = 1.0*tf ; t0 = 0.
   
   test_net, x_exact, xTest,  xdot_exact, xdotTest = test(model, X0, nTest, t_max_test)
   printGroundThruth(test_net, x_exact, xTest,  xdot_exact, xdotTest)


