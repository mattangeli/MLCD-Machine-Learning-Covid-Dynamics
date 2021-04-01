import numpy as np
import time
import copy
from utils import * 
from os import path
from numpy.random import uniform
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses import *
from torch.utils.data import DataLoader

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
        
    torch.save({'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_epoch_loss},
                 PATH)

    time_final  = time.time()
    run_time = time_final - time_initial      
    
    return best_model, Loss_history, run_time




   


