import numpy as np
import torch

# A two hidden layer NN, 1 input & 1 output
class odeNet(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(odeNet,self).__init__()

        # Define the Activation
        self.actF = torch.nn.Sigmoid()
#         self.actF = mySin()

        # define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 1)


    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)
        # output layer
        netOut = self.Lin_out(h)
        return netOut


