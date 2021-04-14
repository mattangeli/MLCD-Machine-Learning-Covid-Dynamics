#!/usr/bin/env python
# coding: utf-8
from training import fit_data_synthetic
from inout import *
from test import *
from utils import  *
from network import saivrNet
from parameters import *
import sys
import os

if __name__ == '__main__':

   ROOT_DIR = checkfolders()

   data_type = 'synthetic' 
  
   t_0 = 0.
   t_final = 31.
   n_timesteps = 100

   # Neural network and optimizer parameters
   layers, hidden_units, activation = 5, 48, 'Sigmoid'
   epochs = 5000
   loss_threshold = 1.e-8
   input_dim, output_dim = 8, 5
   adam_betas, lr = [0.9, 0.999], 8e-4

   # Initializing the neural network and optimizer 
   model = saivrNet(input_dim, layers, hidden_units, output_dim, activation)
   PATH = ROOT_DIR + '/trained_models/{}'.format(model_name)
   
   try:
       # It tries to load the model, otherwise it trains it
       checkpoint = torch.load(PATH)
       model.load_state_dict(checkpoint['model_state_dict'])
       print('The model is trained starting with the weights found in ' + PATH)  
   except FileNotFoundError:
       print('\n File not found ERROR \n' + 
             'NO unsupervised weights found in {} \n'. format(PATH) +
             'Please run unsupervised_main.py first')  
       sys.exit()       
       
   time_series_dict = generate_synthetic_data(model, t_0, t_final, initial_conditions_set, parameters_bundle, n_timesteps) 
   print('\n Fitting to synthetic data')
      
   model, train_losses, optimized_params = fit_data_synthetic(model, time_series_dict, initial_conditions_set, parameters_bundle, parameters_fixed, lr, epochs, loss_threshold)
   
   test_fitmodel(model, data_type, time_series_dict, optimized_params)    
   
   
   
   
   
   
   
                                                           
