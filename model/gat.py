import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv

from model.basic_gnn import GraphSAGE, GCN


class GATNet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, input_heads, intermediate_heads, output_heads, dropout):
        super(GATNet, self).__init__()

        self.gc1 = GATConv(in_channels, nf, heads=input_heads, dropout=dropout)
        self.gc2 = GATConv(nf * input_heads, nf * 2, heads=intermediate_heads, dropout=dropout)
        self.gc3 = GATConv(nf * 2 * intermediate_heads, nf * 4, heads=intermediate_heads, dropout=dropout)
        self.gc4 = GATConv(nf * 4 * intermediate_heads, nf * 4, heads=intermediate_heads, dropout=dropout)
        self.gc5 = GATConv(nf * 4 * intermediate_heads, nf * 4, heads=intermediate_heads, dropout=dropout)
        self.gc6 = GATConv(nf * 4 * intermediate_heads, nf * 2, heads=intermediate_heads, dropout=dropout)
        self.gc7 = GATConv(nf * 2 * intermediate_heads, out_channels, heads=output_heads, dropout=dropout)
        self.dropout = dropout
        self.tanh = torch.nn.Tanh()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()
        self.gc4.reset_parameters()
        self.gc5.reset_parameters()
        self.gc6.reset_parameters()
        self.gc7.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc3(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc4(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc5(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc6(x, edge_index)
        x = F.leaky_relu(x, 0.02)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc7(x, edge_index)

        return self.tanh(x) * 0.5


class GraphSAGENet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GraphSAGENet, self).__init__()
        self.sage = GraphSAGE(in_channels, nf, 7, dropout, torch.nn.LeakyReLU(0.02), aggr='max')
        self.out = SAGEConv(nf, out_channels, aggr='max')
        self.tanh = torch.nn.Tanh()
        self.dropout = dropout

    def reset_parameters(self):
        self.sage.reset_parameters()
        self.out.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage(x, edge_index)

        x = F.leaky_relu(x, 0.02)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x, edge_index)

        return self.tanh(x) * 0.5


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(MLP, self).__init__()
        self.sage = nn.Sequential(
            nn.Linear(in_channels, nf),
            nn.ReLU(),

            nn.Linear(nf, nf),
            nn.ReLU(),

            nn.Linear(nf, nf),
            nn.ReLU(),

            nn.Linear(nf, nf),
            nn.ReLU(),

            nn.Linear(nf, nf),
            nn.ReLU(),

            nn.Linear(nf, nf),
        )
        self.out = nn.Linear(nf, out_channels)
        self.tanh = torch.nn.Tanh()
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        x = self.sage(x)
        x = F.leaky_relu(x, 0.02)
        x = self.out(x)
        return self.tanh(x) * 0.5


class GCNNet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GCNNet, self).__init__()
        self.gcn = GCN(in_channels, nf, 7, dropout, torch.nn.LeakyReLU(0.02))
        self.out = GCNConv(nf, out_channels)
        self.tanh = torch.nn.Tanh()
        self.dropout = dropout

    def reset_parameters(self):
        self.gcn.reset_parameters()
        self.out.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)

        x = F.leaky_relu(x, 0.02)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x, edge_index)

        return self.tanh(x) * 0.5


if __name__ == "__main__":
    print(GATNet(7, 3, 24, 4, 8, 1, 0))
