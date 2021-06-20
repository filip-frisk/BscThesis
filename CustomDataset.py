# imports
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from parseData import parseData

"""
Class name: CustomDataset 
Function: Implements a Dataset object to be able to use DataLoader in PyTorch. 
To do this we need to implement the methods __getitem__ and __len__.
Source: https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
"""

class CustomDataset(Dataset):
    #
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    # Takes an index and returns a tuple (x,y)
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x.cpu())
            if torch.cuda.is_available():
                x = x.to(device=torch.device('cuda'))
        y = self.tensors[1][index]
        return x, y

    # Returns the size of the data
    def __len__(self):
        return self.tensors[0].size(0)
