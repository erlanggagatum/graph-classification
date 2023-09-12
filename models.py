import torch
import torch_geometric

from torch.nn import Linear
import torch.nn.functional as F

# Graph neural network models
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv

# pooling method (for readout layer)
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool

# GCN MODEL CONSTRUCTION
class GCN(torch.nn.Module):
  def __init__(self, dataset, hidden_channels):
    super(GCN, self).__init__()
    
    # weight seed
    torch.manual_seed(42)
    self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.conv3 = GCNConv(hidden_channels, hidden_channels)
    self.lin = Linear(hidden_channels, dataset.num_classes) # for final classification

  def forward(self, x, edge_index, batch):
    # step 1. get node embedding using GCNConv layer
    x = self.conv1(x, edge_index)
    x = x.relu() # apply relu activation after conv
    x = self.conv2(x, edge_index)
    x = x.relu()
    x = self.conv3(x, edge_index)

    # step 2. add readout layer to aggregate all node features of graph
    x = global_mean_pool(x, batch)

    # apply classifier (using linear)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)

    return x
  
# GNN with GraphConv
class GNN(torch.nn.Module):
  def __init__(self, dataset, hidden_channels):
    super(GNN, self).__init__()

    # weight seed
    torch.manual_seed(42)
    self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
    self.conv2 = GraphConv(hidden_channels, hidden_channels)
    self.conv3 = GraphConv(hidden_channels, hidden_channels)
    self.lin = Linear(hidden_channels, hidden_channels)

  def forward(self, x, edge_index, batch):
    # step 1. get node embedding and apply relu activation on each layer
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = self.conv2(x, edge_index)
    x = x.relu()
    x = self.conv3(x, edge_index)

    # step 2. add readout layer to aggregate node features data
    x = global_mean_pool(x, batch)

    # apply classifier
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)

    return x
  
  
# GAT
class GAT(torch.nn.Module):
  def __init__(self, dataset, hidden_channels):
    super(GAT, self).__init__()

    # weight seed
    torch.manual_seed(42)
    self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
    self.conv2 = GATConv(hidden_channels, hidden_channels)
    self.conv3 = GATConv(hidden_channels, hidden_channels)
    self.lin = Linear(hidden_channels, hidden_channels)

  def forward(self, x, edge_index, batch):
    # step 1. get node embedding and apply relu activation on each layer
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = self.conv2(x, edge_index)
    x = x.relu()
    x = self.conv3(x, edge_index)

    # step 2. add readout layer to aggregate node features data
    x = global_mean_pool(x, batch)

    # apply classifier
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)

    return x