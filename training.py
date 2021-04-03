import numpy as np
import time
import copy
from utils import * 
from numpy.random import uniform
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses import *
from torch.utils.data import DataLoader
from random import shuffle

def fit_data(model, time_series_dict, initial_conditions_set, parameters_bundle, lr, epochs,
            loss_threshold):
    
    betas, gammas = parameters_bundle[0][:], parameters_bundle[1][:]
    
    # Train mode
    model.train()

    # Initialize losses arrays
    loss_history, min_loss = [], float('inf')
    
    # handle the data 
    time_sequence = copy.deepcopy(list(time_series_dict.keys()))
    t_0 = time_sequence[0]

    # Initialize  the bundles
    i_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
    r_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=1)

    beta = uniform(betas[0], betas[1], size=1)
    gamma = uniform(gammas[0], gammas[1], size=1)

    i_0 = torch.Tensor([i_0]).reshape((-1, 1))
    r_0 = torch.Tensor([r_0]).reshape((-1, 1))
    beta = torch.Tensor([beta]).reshape((-1, 1))
    gamma = torch.Tensor([gamma]).reshape((-1, 1))

    optimizer = torch.optim.Adam([i_0, r_0, beta, gamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, cooldown = 20, min_lr = 1e-7,  verbose=True)

    i_0.requires_grad = True
    r_0.requires_grad = True
    beta.requires_grad = True
    gamma.requires_grad = True

    s_0 = 1 - (i_0 + r_0)
    initial_conditions = [s_0, i_0, r_0]
    param_bundle = [beta, gamma]
    print('Initial parameters \n' 'S0 = {}, I0 = {}, R0 = {} \n'
          'Beta = {}, Gamma = {}'.format(s_0.item(), i_0.item(),
                                  r_0.item(), beta.item(), gamma.item()))

    
    for epoch in tqdm(range(epochs), desc='Finding the best parameters/initial conditions'):  
        #optimizer.zero_grad()
        epoch_loss = 0.
        batch_loss = 0.  
        shuffle(time_sequence)

        for t in time_sequence:
            true_pop = time_series_dict[t]
            t_tensor = torch.Tensor([t]).reshape(-1,1)

            s_hat, i_hat, r_hat = model.parametric_solution(t_tensor, t_0, initial_conditions, param_bundle)
    
            loss_s, loss_i, loss_r = data_fitting_loss(t_tensor, true_pop, s_hat, i_hat, r_hat)
    
            batch_loss += loss_s + loss_i + loss_r 
    
        batch_loss = batch_loss / len(time_sequence)
        epoch_loss += batch_loss

        batch_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        s_0 = 1 - (i_0 + r_0)
        initial_conditions = [s_0, i_0, r_0]


        if epoch%100 == 0:
            print('Current Loss = {}'.format(epoch_loss.item()), 'lr = ' + str(optimizer.param_groups[0]['lr']))
            print('S0 = {}, I0 = {}, R0 = {} \n'
                  'Beta = {}, Gamma = {}'.format(s_0.item(), i_0.item(),
                                                             r_0.item(), beta.item(),
                                                             gamma.item()))
        if epoch_loss < min_loss:
            min_loss = epoch_loss             
            best_model = copy.deepcopy(model)
            
        if epoch_loss < loss_threshold:
           print('Reached minimum loss')           
           break
           
    return best_model, loss_history, s_0.item(), i_0.item(), r_0.item(), beta.item(), gamma.item()
    


def train_sirNet(model, optimizer, scheduler, t_0, t_final, initial_conditions_set, parameters_bundle, epochs,
               train_size, num_batches, hack_trivial, decay, PATH, loss_threshold = float('-inf')):

    betas, gammas = parameters_bundle[0][:], parameters_bundle[1][:]

   # Train mode
    model.train()

    # Initialize losses arrays
    Loss_history, min_loss = [], 1.
    
    time_initial = time.time()
    ## Training 
    for epoch in tqdm(range(epochs), desc='Training'):
        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(t_0, t_final, train_size, batch_size, perturb=True)
   
        train_epoch_loss = 0.  

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly initial conditions, beta and gamma
            i_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=batch_size)
            r_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=batch_size)
            beta = uniform(betas[0], betas[1], size=batch_size)
            gamma = uniform(gammas[0], gammas[1], size=batch_size)

            i_0 = torch.Tensor([i_0]).reshape((-1, 1))
            r_0 = torch.Tensor([r_0]).reshape((-1, 1))
            beta = torch.Tensor([beta]).reshape((-1, 1))
            gamma = torch.Tensor([gamma]).reshape((-1, 1))

            s_0 = 1 - (i_0 + r_0)
            initial_conditions = [s_0, i_0, r_0]
            param_bundle = [beta, gamma]

            #  Network solutions
            s, i, r = model.parametric_solution(t, t_0, initial_conditions, param_bundle)
            batch_loss = sir_loss(t, s, i, r, param_bundle, decay)
            
            if hack_trivial:
                batch_trivial_loss = trivial_loss(i, hack_trivial)
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward(retain_graph = False)
            optimizer.step()  
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()            
               
        # Keep the loss function history
        Loss_history.append(train_epoch_loss)
        scheduler.step(train_epoch_loss)
        
        if train_epoch_loss < min_loss:
            min_loss = train_epoch_loss
            best_model = copy.deepcopy(model)
        
        if train_epoch_loss < loss_threshold:
            torch.save({'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_epoch_loss},
                         PATH)
            break

        if epoch%250 == 0:
           print('Loss = ' + str(train_epoch_loss), 'lr = ' + str(optimizer.param_groups[0]['lr']))   
        
        if epoch % 1000 == 0 or epoch == epochs -1:
           torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       PATH )
    time_final  = time.time()
    run_time = time_final - time_initial      
    
    return best_model, Loss_history, run_time




   


