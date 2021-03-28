import numpy as np
import torch

# A two hidden layer NN, 1 input & 1 output
class odeNet(torch.nn.Module):
    def __init__(self, input_dim, layers, hidden_units, activation):
        self.layers = layers
        self.D_hid = hidden_units
        self.activation = activation
        self.input_dim = input_dim
        
        super(odeNet,self).__init__()
        # activation
        self.actF = getattr(torch.nn, activation)()

        # layers
        self.Lin_in   = torch.nn.Linear(self.input_dim, self.D_hid)
        self.Lin_nn   = torch.nn.Linear(self.D_hid, self.D_hid)
        self.Lin_out  = torch.nn.Linear(self.D_hid, 1)


    def forward(self,x):
        l = self.Lin_in(x)
        h = self.actF(l)
        
        for layer in range(2, self.layers):
            l = self.Lin_nn(h)
            h = self.actF(l)
        
        netOut = self.Lin_out(h)
        return netOut


