#!/usr/bin/env python
# coding: utf-8
# Author: Mattia Angeli 4/6/21
from training import train_saivrNet
from inout import *
from utils import  *
from network import saivrNet
from parameters import *
import os
import shutil

if __name__ == '__main__':

   ROOT_DIR = checkfolders()
   
   # Time interval
   t_0 = 0.
   t_final = 21.
   
   # Neural network and optimizer parameters
   layers, hidden_units, activation = 5, 50, 'Sigmoid'
   train_size, epochs, n_test = 1000, 5000, 1000
   num_batches, loss_threshold, decay  = 10, 1.e-8, 1e-3
   input_dim, output_dim = 8, 5
   adam_betas, lr = [0.9, 0.999], 5e-3    
   load_weights, hack_trivial = True, 1


   # Initialize the model and optimizer
   model = saivrNet(input_dim, layers, hidden_units, output_dim, activation)
   optimizer = torch.optim.Adam(model.parameters(), lr = lr)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=250, cooldown = 350, min_lr = 1e-6,  verbose=True)
   model_name = 'Unsupervised_a_0={}_i_0={}_v_0={}_r_0={}_beta_1s={}_gammas={}_alpha_1s={}.pt'.format(a_0_set, i_0_set, v_0_set, r_0_set, beta_1s, gammas, alpha_1s)
   PATH = ROOT_DIR + '/trained_models/{}'.format(model_name)

   try:
   # Try loading weights if PATH file exists
       if load_weights:
          checkpoint = torch.load(PATH)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          model.train();
          print('The model is trained starting with the weights found in ' + PATH)
       else:  
          print('The model is trained from scratch')
   except FileNotFoundError:       
          print('File not found in ' + PATH + '\n' + 'The model is trained from scratch')
        
   # Train the model
   model, loss_history, run_time = train_saivrNet(model, optimizer, scheduler, t_0, t_final, initial_conditions_set, parameters_bundle, parameters_fixed, epochs, train_size, num_batches, hack_trivial, decay, model_name, ROOT_DIR, loss_threshold)
                                 
   # Print the loss history and test the NN predictions
   printLoss(loss_history, run_time, model_name)   
   test_model(model, parameters_fixed, t_0 = t_0, t_final=t_final, a0=0.003, i0=0.003, v0=0., r0=0.05, alpha_1=0.24, beta_1=0.25, gamma=0.1)
   
   
   
   
   
