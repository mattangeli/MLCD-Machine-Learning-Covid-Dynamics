
# Fixed parameters
alpha_2 = 0.01
beta_2 = 0.
delta = 1e-2
lam = 0.95
eps = 1/28.
zeta = 1e-3
eta = 5e-4

# Model parameters bundle
a_0_set = [0.1, 0.3]
i_0_set = [0.1, 0.3]
v_0_set = [0., 0.]
r_0_set = [0.1, 0.2]
alpha_1s = [0.15, 0.3]
beta_1s = [0.15, 0.3]
gammas = [0.05, 0.15]

# set the initial conditions bundle
initial_conditions_set = [a_0_set, i_0_set, v_0_set, r_0_set]

# set the parameters bundle
parameters_bundle = [alpha_1s, beta_1s, gammas]

# set the fixed parameters dictionary
parameters_fixed = { 'alpha_2' : alpha_2,
                        'beta_2' : beta_2, 
                        'delta' : delta, 
                        'lam' : lam, 
                        'eps' : eps, 
                        'zeta' : zeta, 
                        'eta' : eta }
