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

   ROOT_DIR = checkfolders()   # create the folders needed to run the calculation
   
   # Time interval (days)
   t_0 = 0.
   t_final = 31.
   
   # Neural network and optimizer parameters
   layers, hidden_units, activation = 5, 48, 'Sigmoid'  
   train_size, epochs = 1024, 7500
   num_batches, loss_threshold,  = 8, 1.e-6
   hack_trivial = 1   #prevent the network from solving the model trivially
   decay = 1e-4  # exponential regularization that favors the first points
   input_dim, output_dim = 8, 5
   adam_betas, lr = [0.9, 0.999], 5e-3   
   load_weights = True

   # Initialize the model and optimizer
   model = saivrNet(input_dim, layers, hidden_units, output_dim, activation)
   optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = adam_betas)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=550, 
                                                         cooldown = 550, min_lr = 1e-6, 
                                                          verbose=True)
   PATH = ROOT_DIR + '/trained_models/{}'.format(model_name)

   try:
   # Try loading weights if PATH file exists
       if load_weights:
          checkpoint = torch.load(PATH)
          model.load_state_dict(checkpoint['model_state_dict'])
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
   test_model(model,initial_conditions_set, parameters_bundle, parameters_fixed, t_0 = t_0, t_final=t_final)
   
   
   
   
   
