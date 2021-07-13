import torch 
import torch.nn as nn 


class Dummy(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Dummy, self).__init__()
        self.fc1 =  nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return x