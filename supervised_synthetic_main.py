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
   t_final = 50.
   n_timesteps = 100

   # Neural network and optimizer parameters
   layers, hidden_units, activation = 5, 50, 'Sigmoid'
   epochs = 5000
   loss_threshold = 1.e-8
   input_dim, output_dim = 8, 5
   adam_betas, lr = [0.9, 0.999], 5e-4

   # Initializing the neural network and optimizer 
   model = saivrNet(input_dim, layers, hidden_units, output_dim, activation)
   model_name = 'Unsupervised_a_0={}_i_0={}_v_0={}_r_0={}_beta_1s={}_gammas={}_alpha_1s={}.pt'.format(a_0_set, i_0_set, v_0_set, r_0_set, beta_1s, gammas, alpha_1s)
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
      
   model, train_losses, s_0_fit, a_0_fit, i_0_fit, v_0_fit, r_0_fit, alpha_1_fit, beta_1_fit, gamma_fit = fit_data_synthetic(model, time_series_dict, initial_conditions_set, parameters_bundle, parameters_fixed, lr, epochs, loss_threshold)
   
   print('\n Fitted initial conditions: S0 = {:.2f}, A0 = {:.2f}, I0 = {:.2f}, V0 = {:.2f}, R0 = {:.2f} \n'
          ' Fitted Alpha_1 = {:.2f}, Beta_1 = {:.2f}, Gamma = {:.2f} \n'.format(s_0_fit, a_0_fit, i_0_fit, v_0_fit,
                                                             r_0_fit, alpha_1_fit, beta_1_fit,
                                                             gamma_fit))  
   
   optimized_params = s_0_fit, a_0_fit, i_0_fit, v_0_fit, r_0_fit, alpha_1_fit, beta_1_fit, gamma_fit                                                          
   test_fitmodel(model, data_type, time_series_dict, optimized_params)    
   
   
   
   
   
   
   
                                                           
