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





# unsupervised training
def train_saivrNet(model, optimizer, scheduler, t_0, t_final, initial_conditions_set, parameters_bundle, parameters_fixed, epochs,
                   train_size, num_batches, hack_trivial, decay, weight_i_loss,
                   model_name, ROOT_DIR, loss_threshold = float('-inf')):

    alpha_1s, beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:], parameters_bundle[3][:]

   # Train mode
    model.train()             
    best_model = copy.deepcopy(model)

    # Initialize losses arrays
    Loss_history, min_loss = [], 1.
    
    time_initial = time.time()

    for epoch in tqdm(range(epochs), desc='Training'):
        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(t_0, t_final, train_size, batch_size, perturb=True)
   
        train_epoch_loss = 0.  

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly initial conditions, alpha_1, beta_1 and gamma
            a_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=batch_size)
            i_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=batch_size)
            v_0 = uniform(initial_conditions_set[2][0], initial_conditions_set[2][1], size=batch_size)
            r_0 = uniform(initial_conditions_set[3][0], initial_conditions_set[3][1], size=batch_size)
            alpha_1 = uniform(alpha_1s[0], alpha_1s[1], size=batch_size)
            beta_1 = uniform(beta_1s[0], beta_1s[1], size=batch_size)
            gamma = uniform(gammas[0], gammas[1], size=batch_size)
            delta = uniform(deltas[0], deltas[1], size=batch_size)
            
            a_0 = torch.Tensor([a_0]).reshape((-1, 1))
            i_0 = torch.Tensor([i_0]).reshape((-1, 1))
            v_0 = torch.Tensor([v_0]).reshape((-1, 1))
            r_0 = torch.Tensor([r_0]).reshape((-1, 1))
            alpha_1 = torch.Tensor([alpha_1]).reshape((-1, 1))
            beta_1 = torch.Tensor([beta_1]).reshape((-1, 1))
            gamma = torch.Tensor([gamma]).reshape((-1, 1))
            delta = torch.Tensor([delta]).reshape((-1, 1))
            
            s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
            initial_conditions = [s_0, a_0, i_0, v_0, r_0]
            param_bundle = [alpha_1, beta_1, gamma, delta]

            # Compute Network solution and loss
            s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, param_bundle)
            batch_loss = sir_loss(t, s, a, i, v, r, param_bundle, parameters_fixed, decay, weight_i_loss)
            
            # Hacking the system to prevent the ODEs to be solved trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(i, a, v, hack_trivial)
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward(retain_graph = False)
            optimizer.step()  
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()            
               
        # Keep the loss function history
        train_epoch_loss = train_epoch_loss / batch_size
        Loss_history.append(train_epoch_loss)
        scheduler.step(train_epoch_loss)
        
        if train_epoch_loss < min_loss:
            min_loss = train_epoch_loss
            best_model = copy.deepcopy(model)
        
        # stop the training if the loss is lower than the threshold
        if train_epoch_loss < loss_threshold:
            torch.save({'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_epoch_loss},
                         ROOT_DIR + '/trained_models/{}'.format(model_name))
            break
            
        # print the loss and a plot of one of the solutions in the Checkpoint folder
        if epoch%250 == 0:
           print('Loss = ' + str(train_epoch_loss), 'lr = ' + str(optimizer.param_groups[0]['lr']))   
           test_snippet(model, epoch, train_epoch_loss, t_0, t_final, parameters_fixed, a0 = a_0[0] , i0=i_0[0], v0=v_0[0], 
                                                                           r0=r_0[0], alpha_1 = alpha_1[0], 
                                                                           beta_1 = beta_1[0], gamma = gamma[0],
                                                                           delta = delta[0])
        
        if epoch % 250 == 0 or epoch == epochs -1:
           torch.save({'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ROOT_DIR + '/trained_models/{}'.format(model_name) )
    time_final  = time.time()
    run_time = time_final - time_initial      
    
    return best_model, Loss_history, run_time







def fit_data_synthetic(model, time_series_dict, initial_conditions_set, parameters_bundle, param_fixed, lr, epochs,
            loss_threshold):
    
    alpha_1s, beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:], parameters_bundle[3][:]
    
    # Train mode
    model.train()

    # Initialize losses arrays
    loss_history, min_loss = [], float('inf')
    
    # handle the data 
    time_sequence = copy.deepcopy(list(time_series_dict.keys()))
    t_0 = time_sequence[0]

    # Initialize  the bundles
    a_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
    i_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=1)
    v_0 = uniform(initial_conditions_set[2][0], initial_conditions_set[2][1], size=1)
    r_0 = uniform(initial_conditions_set[3][0], initial_conditions_set[3][1], size=1)
    alpha_1 = uniform(alpha_1s[0], alpha_1s[1], size=1)
    beta_1 = uniform(beta_1s[0], beta_1s[1], size=1)
    gamma = uniform(gammas[0], gammas[1], size=1)
    delta = uniform(deltas[0], deltas[1], size=1)
    
    a_0 = torch.Tensor([a_0]).reshape((-1, 1))
    i_0 = torch.Tensor([i_0]).reshape((-1, 1))
    v_0 = torch.Tensor([v_0]).reshape((-1, 1))
    r_0 = torch.Tensor([r_0]).reshape((-1, 1))
    alpha_1 = torch.Tensor([alpha_1]).reshape((-1, 1))
    beta_1 = torch.Tensor([beta_1]).reshape((-1, 1))
    gamma = torch.Tensor([gamma]).reshape((-1, 1))
    delta = torch.Tensor([delta]).reshape((-1, 1))
        
    a_0.requires_grad = True
    i_0.requires_grad = True
    v_0.requires_grad = True
    r_0.requires_grad = True
    beta_1.requires_grad = True
    alpha_1.requires_grad = True    
    gamma.requires_grad = True
    delta.requires_grad = True

    s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
    initial_conditions = [s_0, a_0, i_0, v_0, r_0]
    param_bundle = [alpha_1, beta_1, gamma, delta]

    print('\n Initial (random) parameters \n' 'S0 = {:.2f}, A0 = {:.2e}, I0 = {:.2e}, V0 = {:.2f}, R0 = {:.2f} \n'
          'Alpha_1 = {:.2f}, Beta_1 = {:.2f}, $\gamma$ = {:.2f}, $\delta$ = {:.2e} \n'.format(s_0.item(), a_0.item(), i_0.item(),
                                                                                 v_0.item(), r_0.item(), alpha_1.item(),
                                                                                 beta_1.item(), gamma.item(), delta.item()))

    optimizer = torch.optim.Adam([a_0, i_0, v_0, r_0, alpha_1, beta_1, gamma, delta], lr=lr)
    
    for epoch in tqdm(range(epochs), desc='Finding the best parameters/initial conditions'):  
        epoch_loss = 0.
        batch_loss = 0.  
        shuffle(time_sequence)

        for t in time_sequence:
            true_pop = time_series_dict[t]
            t_tensor = torch.Tensor([t]).reshape(-1,1)

            s_hat, a_hat, i_hat, v_hat, r_hat = model.parametric_solution(t_tensor, t_0,
                                                                          initial_conditions,
                                                                               param_bundle)
            
            losses = data_fitting_loss(t_tensor, true_pop, s_hat, a_hat, i_hat, v_hat, r_hat)
            batch_loss += sum(losses)
                              
        batch_loss = batch_loss / len(time_sequence)
        epoch_loss += batch_loss

        batch_loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(epoch_loss)
        
        s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
        initial_conditions = [s_0, a_0, i_0, v_0, r_0]
    
        if epoch%250 == 0 and epoch != 0:
            print('Current Loss = {}'.format(epoch_loss.item()), 'lr = ' 
                                      + str(optimizer.param_groups[0]['lr']))
            print('Fitted parameters \n' 'S0 = {:.2f}, A0 = {:.2e}, I0 = {:.2e}, V0 = {:.2f}, R0 = {:.2f} \n'
                                       'Alpha_1 ={:.2f}, Beta_1 = {:.2f}, $\gamma$ = {:.2f}, $\delta$ = {:.2e} \n'
                                       .format(s_0.item(), a_0.item(), i_0.item(), v_0.item(),
                                        r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
          
        if epoch_loss < min_loss:
            min_loss = epoch_loss             
            best_model = copy.deepcopy(model)
            
        if epoch_loss < loss_threshold:
           print('\n Reached minimum loss \n')           
           break
    
    optimized_params = [ s_0.item(), a_0.item(), i_0.item(), v_0.item(),
                        r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()]       
    return best_model, loss_history, optimized_params
    




def fit_data_real(model, time_series_dict, initial_conditions_set, parameters_bundle, param_fixed, lr, epochs,
            loss_threshold):
    
    alpha_1s, beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:], parameters_bundle[3][:]

    # handle the data 
    time_sequence = copy.deepcopy(list(time_series_dict.keys()))
    t_0 = time_sequence[0]

    # Initialize losses arrays
    loss_history, min_loss = [], float('inf')
        
    # Train mode
    model.train()
    
    # Initialize  the bundles
    a_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
    i_0 = time_series_dict[t_0][1]
    v_0 = time_series_dict[t_0][2]
    r_0 = time_series_dict[t_0][3]
    alpha_1 = uniform(alpha_1s[0], alpha_1s[1], size=1)
    beta_1 = uniform(beta_1s[0], beta_1s[1], size=1)
    gamma = uniform(gammas[0], gammas[1], size=1)
    delta = 0.#uniform(deltas[0], deltas[1], size=1)
    
    a_0 = torch.Tensor([a_0]).reshape((-1, 1))
    i_0 = torch.Tensor([i_0]).reshape((-1, 1))
    v_0 = torch.Tensor([v_0]).reshape((-1, 1))
    r_0 = torch.Tensor([r_0]).reshape((-1, 1))
    alpha_1 = torch.Tensor([alpha_1]).reshape((-1, 1))
    beta_1 = torch.Tensor([beta_1]).reshape((-1, 1))
    gamma = torch.Tensor([gamma]).reshape((-1, 1))
    delta = torch.Tensor([delta]).reshape((-1, 1))
        
    a_0.requires_grad = True
    i_0.requires_grad = True
    r_0.requires_grad = True
    v_0.requires_grad = False
    beta_1.requires_grad = True
    alpha_1.requires_grad = True    
    gamma.requires_grad = True
    delta.requires_grad = False

    s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
    initial_conditions = [s_0, a_0, i_0, v_0, r_0]
    param_bundle = [alpha_1, beta_1, gamma, delta]

    print('\n Initial parameters \n' 'S0 = {:.2f}, A0 = {:.2e}, I0 = {:.2e}, V0 = {:.2e}, R0 = {:.2f} \n'
          'Alpha_1 = {:.2f}, Beta_1 = {:.2f}, $\gamma$ = {:.2f}, $\delta$ = {:.2f} \n'.format(s_0.item(), a_0.item(), i_0.item(),
                                     v_0.item(), r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))

    learning_rate_dicts = [ {'params' : [i_0, v_0], 'lr' : 1e-5},
                            {'params' : [alpha_1, beta_1, gamma, r_0, a_0], 'lr' : 1e-1},
                            {'params' : [delta], 'lr' : 1e-3} ]
                            
                            
    optimizer = torch.optim.SGD(learning_rate_dicts, lr=lr)
    
    for epoch in tqdm(range(epochs), desc='Finding the best parameters/initial conditions'):  
        epoch_loss = 0.
        batch_loss = 0.  
        shuffle(time_sequence)

        for t in time_sequence:
            true_pop = time_series_dict[t]
            t_tensor = torch.Tensor([t]).reshape(-1,1)
            t_tensor.requires_grad = True

            s_hat, a_hat, i_hat, v_hat, r_hat = model.parametric_solution(t_tensor, t_0,
                                                                          initial_conditions,
                                                                               param_bundle)
            
            loss_i = data_fitting_loss(t_tensor, true_pop, s_hat, a_hat, i_hat, v_hat, r_hat, mode = 'mse', data_type = 'real')
                        
            # regularization to avoid negative asymptomatic population in extreme cases
            #regularization = torch.exp(-0.005 * i_0)  + torch.exp(-0.005 * a_0)

            batch_loss += loss_i # + regularization
            
        batch_loss = batch_loss / len(time_sequence)
        epoch_loss += batch_loss

        batch_loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(epoch_loss)
        
        s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
        initial_conditions = [s_0, a_0, i_0, v_0, r_0]
        
        optimized_params = s_0.item(), a_0.item(), i_0.item(), v_0.item(), r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()                                                        
    
        if epoch%250 == 0 and epoch != 0:
            print('Current Loss = {}'.format(epoch_loss.item()), 'lr = ' 
                                      + str(optimizer.param_groups[0]['lr']))
            print('Fitted parameters \n' 'S0 = {:.2f}, A0 = {:.2e}, I0 = {:.2e}, V0 = {:.2e}, R0 = {:.2f} \n'
                                       'Alpha_1 ={:.2f}, Beta_1 = {:.2f}, $\gamma$ = {:.2f}, $\delta$ = {:.2e} \n'
                                       .format(s_0.item(), a_0.item(), i_0.item(), v_0.item(),
                                        r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
            test_fitmodel_checkpoint(model, epoch, time_series_dict, 
                                                optimized_params, param_fixed)  
                                                
        if epoch_loss < min_loss:
            min_loss = epoch_loss             
            best_model = copy.deepcopy(model)
            
        if epoch_loss < loss_threshold:
           print('\n Reached minimum loss \n')           
           break
           
    return best_model, loss_history, optimized_params
    




