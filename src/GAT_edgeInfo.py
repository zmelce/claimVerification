import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops  #, softmax
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter
import math
from torch_geometric.datasets import ZINC
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def softmax(src, index, num_nodes):

    N = int(index.max()) + 1 if num_nodes is None else num_nodes

    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]

    return out / (out_sum + 1e-16)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class GATedgeAttr(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_dim,
                 heads=2,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(GATedgeAttr, self).__init__(node_dim=0, aggr='add') 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))    # emb(in) x [H*emb(out)]
        #self.weight = torch.nn.init.xavier_uniform(self.weight)

        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))   # 1 x H x [2*emb(out)+edge_dim]
        self.edge_updates = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))   # [emb(out)+edge_dim] x emb(out)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_updates)  # new
        zeros(self.bias)


    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)   # N x H x emb(out)
        if size is None and torch.is_tensor(x):
            #edge_index, _ = remove_self_loops(edge_index)   # 2 x E
            #print('edge_index', edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # 2 x (E+N)


        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (E+N) x edge_dim

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):  

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim

        x_j = torch.cat([x_j, edge_attr], dim=-1)  # (E+N) x H x (emb(out)+edge_dim)

        x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        return x_j * alpha.view(-1, self.heads, 1)   # (E+N) x H x (emb(out)+edge_dim)

    def update(self, aggr_out):

        aggr_out = aggr_out.mean(dim=1) # N x (emb(out)+edge_dim)
        aggr_out = torch.mm(aggr_out, self.edge_updates)
      
        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

class GNNStack(nn.Module):
    def __init__(self, in_channels, out_channels,edge_dim,out_dim, task='graph'):
        super(GNNStack, self).__init__()
        self.task = task
        self.att = nn.ModuleList()
        self.att.append(self.build_att_model(in_channels, out_channels,edge_dim))
        #self.att = (self.build_att_model(in_channels, out_channels,edge_dim))
        #self.lns = nn.LayerNorm(out_channels)
        #self.lns.append(nn.LayerNorm(out_channels))
        #self.lns.append(nn.LayerNorm(out_channels))

        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(out_channels))
        self.lns.append(nn.LayerNorm(out_channels))

        for l in range(3):
            self.att.append(self.build_att_model(in_channels, out_channels,edge_dim))

        self.post_mp = nn.Linear(out_channels, out_dim)
        # self.post_mp = nn.Sequential(
        #     nn.Linear(out_channels, out_channels), nn.Dropout(0.5), ##0.25 #0.15
        #     nn.Linear(out_channels, out_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.5 ##0.25 #0.15
        self.num_layers = 3

    def build_att_model(self, in_channels, out_channels,edge_dim):
        if self.task == 'node':
            return GATedgeAttr(in_channels, out_channels,edge_dim)
        else:
            return GATedgeAttr(in_channels, out_channels,edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):

        for i in range(self.num_layers):
          x = self.att[i](x, edge_index,edge_attr)
          emb = x
          x = F.relu(x)
          x = F.dropout(x, p=self.dropout, training=self.training)
          if not i == self.num_layers - 1:
              x = self.lns[i](x)

        if self.task == 'graph':
           x_glob = global_mean_pool(x, batch)

        return x,emb

    def loss(self, pred, label):
        m = nn.Sigmoid()
        loss_func = nn.BCELoss()
        #return loss_func(m(pred), label)
        return nn.CrossEntropyLoss(pred,label)
