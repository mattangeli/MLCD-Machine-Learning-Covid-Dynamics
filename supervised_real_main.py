#!/usr/bin/env python
# coding: utf-8
from training import fit_data_real
from inout import *
from test import *
from utils import  *
from network import saivrNet
from parameters import *
import sys
import os

if __name__ == '__main__':

   ROOT_DIR = checkfolders()

   data_type = 'real' 
   
   country = 'Argentina'
   begin_date = '11/15/20' #month/day/year format
   final_date = '04/20/21'
   recovered_mode = 'retarded' # 'retarded' if the recovered population is not present, 'Data' otherwise
   time_series_dict = get_dataframe(country, begin_date, final_date, recovered_mode, moving_average = False) 
   
   t_0 = 0.
   t_final = len(time_series_dict)
   size = len(time_series_dict)

   # Training parameters
   epochs = 5000
   loss_threshold = 1.e-10
   lr = 1e-2

   # Neural network and optimizer parameters
   model = saivrNet(input_dim, layers, hidden_units, output_dim, activation)
   optimizer = torch.optim.Adam(model.parameters(), lr = lr)
   PATH = ROOT_DIR + '/trained_models/{}'.format(model_name)

   checkpoint = torch.load(PATH)
   model.load_state_dict(checkpoint['model_state_dict'])
   print('The model is trained starting with the weights found in ' + PATH)
   print('\n Fitting real data')
      
   model, train_losses, optimized_params = fit_data_real(model, time_series_dict, initial_conditions_set, parameters_bundle, parameters_fixed, lr, epochs, loss_threshold)
   
   test_fitmodel(model, data_type, time_series_dict, optimized_params, average)    
   test_fitmodel_groudtruth(model, data_type, time_series_dict, optimized_params, parameters_fixed)
   
   
   
   
   
   
                                                           
