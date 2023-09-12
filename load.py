import torch
import os
import torch
from torch_geometric.datasets import TUDataset

def loadTUDataset():
  return TUDataset(root='data/TUDataset', name='MUTAG')

# loadTUDataset()
