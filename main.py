#!/usr/bin/env python
# coding: utf-8
from training import train_sirNet
from inout import *
from test import *
from utils import  *
from network import sirNet
import os

if __name__ == '__main__':

   LOAD_DIR, _ = checkfolders()
   
   country = 'Israel'
   begin_date = '12/1/20' #month/day/year format
   average = 'W' #'D' or 'M' time averages
   infected, removed = get_dataframe(country, begin_date, average)

   t_0 = 0.
   t_final = 50.
   
   i_0_set = [0.1, 0.3]
   r_0_set = [0.1, 0.3]
   betas = [0.2, 0.3]
   gammas = [0.05, 0.15]

    # Model parameters
   initial_conditions_set = [i_0_set, r_0_set]
   parameters_bundle = [betas, gammas]

   # Neural network and optimizer parameters
   layers, hidden_units, activation = 4, 50, 'Sigmoid'
   train_size, epochs, n_test = 1000,5000, 1000
   num_batches, loss_threshold, decay  = 10, 1.e-10, 1.e-3
   input_dim, output_dim = 5, 3
   adam_betas, lr = [0.9, 0.999], 5e-3    
   load_weights, hack_trivial = False, 1


   # Initialize the model and optimizer
   model = sirNet(input_dim, layers, hidden_units, output_dim, activation)
   optimizer = torch.optim.Adam(model.parameters(), lr = lr)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=150, cooldown = 250, min_lr = 1e-5,  verbose=True)
   model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                              betas,
                                                              gammas)
   PATH = LOAD_DIR + model_name

   if load_weights:
       ## Try loading weights if PATH file exists
       checkpoint = torch.load(PATH)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       init_loss = checkpoint['loss']
       model.train();
       print('The model is trained starting with the weights found in ' + PATH)
   else:    
       print('The model is trained from scratch.')       
        
   # train the model
   sir, loss_history, run_time = train_sirNet(model, optimizer, scheduler, t_0, t_final, initial_conditions_set, parameters_bundle, epochs, train_size, num_batches, hack_trivial, decay, PATH,
                                 loss_threshold)
                                 

   printLoss(loss_history, run_time, PATH)
   
   test_model(model, t_0 = t_0, t_final=t_final, i0=0.2, r0=0.1)
   
   
   
   
   
