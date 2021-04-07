import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class saivrNet(torch.nn.Module):
    def __init__(self, input_dim=5, layers=10, hidden=3, output=3, activation=None):
       
        super(saivrNet, self).__init__()
        #activation
        if activation is None:
            self.actF = torch.nn.Sigmoid()
        else:
            self.actF = getattr(torch.nn, activation)()

        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.actF
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, output)
        )

    def forward(self, x):
        x = self.ffn(x)
        s_N = (x[:, 0]).reshape(-1, 1)
        a_N = (x[:, 1]).reshape(-1, 1)
        i_N = (x[:, 2]).reshape(-1, 1)
        v_N = (x[:, 3]).reshape(-1, 1)
        r_N = (x[:, 4]).reshape(-1, 1)
        return s_N, a_N, i_N, v_N, r_N

    def parametric_solution(self, t_tensor, t_0, initial_conditions, param_bundle):
        s_0, a_0, i_0, v_0, r_0 = initial_conditions[0][:], initial_conditions[1][:], initial_conditions[2][:], initial_conditions[3][:] ,initial_conditions[4][:] 
 
        alpha_1, beta_1, gamma = param_bundle[0][:], param_bundle[1][:], param_bundle[2][:]

        dt = t_tensor - t_0
        f = (1 - torch.exp(-dt))
        
        t_bundle = torch.cat([t_tensor, a_0, i_0, v_0, r_0, alpha_1, beta_1, gamma], dim=1)
        
        N = self.forward(t_bundle)
        
        N0, N1, N2, N3, N4 = N

        to_softmax = torch.cat([N0, N1, N2, N3, N4], dim=1)
        softmax_output = softmax(to_softmax, dim=1)
        N0, N1, N2, N3, N4 = softmax_output[:, 0], softmax_output[:, 1], softmax_output[:, 2], softmax_output[:, 3], softmax_output[:, 4]
        N0, N1, N2, N3, N4 = N0.reshape(-1, 1), N1.reshape(-1, 1), N2.reshape(-1, 1), N3.reshape(-1, 1), N4.reshape(-1, 1)

        s_hat = (s_0 + f * (N0 - s_0))
        a_hat = (a_0 + f * (N1 - a_0))
        i_hat = (i_0 + f * (N2 - i_0))
        v_hat = (v_0 + f * (N3 - v_0))
        r_hat = (r_0 + f * (N4 - r_0))

        return s_hat, a_hat, i_hat, v_hat, r_hat


