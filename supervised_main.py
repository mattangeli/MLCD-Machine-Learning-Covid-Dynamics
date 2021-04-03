#!/usr/bin/env python
# coding: utf-8
from training import fit_data
from inout import *
from test import *
from utils import  *
from network import sirNet
import os

if __name__ == '__main__':

   ROOT_DIR, _, _ = checkfolders()

   country = 'Israel'
   begin_date = '12/1/20' #month/day/year format
   average = 'W' #'D' or 'M' time averages
   infected, removed = get_dataframe(country, begin_date, average)

   t_0 = 0.
   t_final = 50.
   size = 150
 
   i_0_set = [0.1, 0.3]
   r_0_set = [0.1, 0.3]
   betas = [0.2, 0.3]
   gammas = [0.05, 0.15]

    # Model parameters
   initial_conditions_set = [i_0_set, r_0_set]
   parameters_bundle = [betas, gammas]

   # Neural network and optimizer parameters
   layers, hidden_units, activation = 4, 50, 'Sigmoid'
   epochs = 2000
   loss_threshold  = 1.e-10
   input_dim, output_dim = 5, 3
   lr = 8e-4   


   # Initialize the model and optimizer
   model = sirNet(input_dim, layers, hidden_units, output_dim, activation)
   model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                              betas, gammas)
   PATH = ROOT_DIR + '/trained_models/unsupervised/{}'.format(model_name)
   checkpoint = torch.load(PATH)
   model.load_state_dict(checkpoint['model_state_dict'])
   print('The model is trained starting with the weights found in ' + PATH)
  
   time_series_dict = generate_synthetic_data(model, t_0, t_final, size) 
   
   model, train_losses, s_0_fit, i_0_fit, r_0_fit, beta_fit, gamma_fit = fit_data(model, time_series_dict, initial_conditions_set, parameters_bundle, lr, epochs, loss_threshold)
   
   print('Estimated initial conditions: S0 = {}, I0 = {}, R0 = {} \n'
          'Estimated Beta = {}, Estimated Gamma = {}'.format(s_0_fit, i_0_fit,
                                                             r_0_fit, beta_fit,
                                                             gamma_fit))  
   
   optimized_params = s_0_fit, i_0_fit, r_0_fit, beta_fit, gamma_fit                                                          
   test_fitmodel(model, time_series_dict, optimized_params, lineW=3)    
   
   
   
   
   
   
   
                                                           
