import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.utils.repeat import repeat
from torch_geometric.nn.models.basic_gnn import GraphSAGE, GCN


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
        self.sage = GraphSAGE(in_channels, nf, 7, out_channels, dropout, torch.nn.LeakyReLU(0.02), aggr='max')
        self.tanh = torch.nn.Tanh()
        self.dropout = dropout

    def reset_parameters(self):
        self.sage.reset_parameters()

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
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


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(SAGEConv(in_channels, channels, aggr='max'))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(SAGEConv(channels, channels, aggr='max'))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(SAGEConv(in_channels, channels, aggr='max'))
        self.up_convs.append(SAGEConv(in_channels, out_channels, aggr='max'))

        self.tanh = torch.nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight=edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index,
                                                                           edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return self.tanh(x) * 0.5

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


if __name__ == "__main__":
    print(GATNet(7, 3, 24, 4, 8, 1, 0))
