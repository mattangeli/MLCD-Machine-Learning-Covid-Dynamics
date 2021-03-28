#!/usr/bin/env python
# coding: utf-8
from training import run_odeNet
from inout import *
from solve import *
from test import *


if __name__ == '__main__':

   checkfolders()
   
   # define the bundles
   t0, tf = 0.,  2
   x0, x0f, x0Test = 1., 2., 1.323677
   lam0, lamf, lamTest = 0.4, 0.8, 0.7521

   X0 = [t0,x0, lam0]
   Xf = [tf, x0f, lamf]
   Xtest = [x0Test, lamTest]
   
   # neural network and optimizer parameters
   layers, hidden_units, activation = 4, 50, 'Tanh'
   n_train, epochs, nTest = 100, int(5e4), 500
   minibatch_number, minLoss, lr  = 1, 1e-10, 8e-4
   betas = [0.9, 0.999]    
   
   # train the model
   model, loss, runTime = run_odeNet(X0, Xf, layers, hidden_units, activation, epochs,
      n_train, lr, betas, minibatch_number, minLoss, loadWeights = False, PATH = 'models/expDE')
   
   printLoss(loss, runTime)
   
   
   test_net, x_exact, xTest,  xdot_exact, xdotTest = test_ode_solution(model, t0, tf, Xtest, nTest)
   printGroundThruth(test_net, x_exact, xTest,  xdot_exact, xdotTest)


   ntTest = 50
   nxTest = 80
   
   Losses = test_ode_solution_bundle(model, X0, Xf, ntTest, nxTest)

   print_scatter(Losses)
