import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.nn import TopKPooling, GCNConv, BatchNorm, InstanceNorm, GraphNorm
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_geometric.utils.repeat import repeat
from torch_geometric.nn.models.basic_gnn import GraphSAGE, GCN, GIN, GAT
from torch_sparse import spspmm


class GATNet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GATNet, self).__init__()

        self.gat_head = GAT(in_channels, nf, 5, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=nn.BatchNorm1d(nf))
        self.gat_tail = GAT(nf * 3, nf, 2, out_channels, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=nn.BatchNorm1d(nf))
        self.skip = GAT(in_channels, nf, 1, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=nn.BatchNorm1d(nf))
        self.global_head = GAT(in_channels, nf, 2, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=nn.BatchNorm1d(nf))
        self.tanh = torch.nn.Tanh()

    def reset_parameters(self):
        self.gat.reset_parameters()

    def forward(self, x, edge_index):
        x_head = self.gat_head(x, edge_index)
        x_skip = self.skip(x, edge_index)
        x_global = self.global_head(x, edge_index).mean(dim=0).unsqueeze(0).expand((x_head.shape[0], -1))
        x_out = self.gat_tail(torch.cat([x_head, x_skip, x_global], dim=1), edge_index)
        return self.tanh(x_out) * 0.5


class GraphSAGENetSkip(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GraphSAGENetSkip, self).__init__()
        self.sage_head = GraphSAGE(in_channels, nf, 7, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=GraphNorm(nf))
        self.skip = GraphSAGE(in_channels, nf, 1, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=GraphNorm(nf))
        self.sage_tail = GraphSAGE(nf * 2, nf, 1, out_channels, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=GraphNorm(nf))
        self.tanh = torch.nn.Tanh()

    def reset_parameters(self):
        self.sage_head.reset_parameters()
        self.sage_tail.reset_parameters()
        self.skip.reset_parameters()

    def forward(self, x, edge_index):
        x_head = self.sage_head(x, edge_index)
        x_skip = self.skip(x, edge_index)
        x_out = self.sage_tail(torch.cat([x_head, x_skip], dim=1), edge_index)
        return self.tanh(x_out) * 0.5


class GraphSAGENet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GraphSAGENet, self).__init__()
        self.sage_head = GraphSAGE(in_channels, nf, 7, out_channels, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=GraphNorm(nf))
        self.tanh = torch.nn.Tanh()

    def reset_parameters(self):
        self.sage_head.reset_parameters()

    def forward(self, x, edge_index, *_args):
        x_out = self.sage_head(x, edge_index)
        return self.tanh(x_out) * 0.5


class DoubleSAGEConv(nn.Module):

    def __init__(self, nf_in, nf_out, norm, activation, skip_norm=False):
        super().__init__()
        self.conv_0 = SAGEConv(nf_in, nf_out, aggr='max')
        self.norm_0 = norm(nf_out)
        self.conv_1 = SAGEConv(nf_out, nf_out, aggr='max')
        self.norm_1 = norm(nf_out)
        self.skip_norm = skip_norm
        self.activation = activation

    def forward(self, x, edge_index):
        x = self.conv_0(x, edge_index)
        if not self.skip_norm:
            x = self.activation(self.norm_0(x))
        else:
            x = self.activation(x)
        x = self.conv_1(x, edge_index)
        if not self.skip_norm:
            x = self.activation(self.norm_1(x))
        else:
            x = self.activation(x)
        return x

    def reset_parameters(self):
        self.conv_0.reset_parameters()
        self.conv_1.reset_parameters()
        self.norm_0.reset_parameters()
        self.norm_1.reset_parameters()


class GraphSAGEEncoderDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, nf):
        super().__init__()
        norm = GraphNorm
        self.layer_0_e = DoubleSAGEConv(in_channels, nf, norm, torch.nn.LeakyReLU(0.02))
        self.layer_1_e = DoubleSAGEConv(nf, nf * 2, norm, torch.nn.LeakyReLU(0.02))
        self.layer_2_e = DoubleSAGEConv(nf * 2, nf * 4, norm, torch.nn.LeakyReLU(0.02))
        self.layer_3_e = DoubleSAGEConv(nf * 4, nf * 8, norm, torch.nn.LeakyReLU(0.02))
        self.layer_4 = DoubleSAGEConv(nf * 8, nf * 8, norm, torch.nn.LeakyReLU(0.02))

        self.layer_3_d = DoubleSAGEConv(nf * 8, nf * 4, norm, torch.nn.LeakyReLU(0.02))
        self.layer_2_d = DoubleSAGEConv(nf * 4, nf * 2, norm, torch.nn.LeakyReLU(0.02))
        self.layer_1_d = DoubleSAGEConv(nf * 2, nf, norm, torch.nn.LeakyReLU(0.02))
        self.layer_0_d = DoubleSAGEConv(nf, nf, norm, torch.nn.LeakyReLU(0.02))
        self.layer_out = nn.Linear(nf, out_channels)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        self.layer_0_e.reset_parameters()
        self.layer_1_e.reset_parameters()
        self.layer_2_e.reset_parameters()
        self.layer_3_e.reset_parameters()
        self.layer_4.reset_parameters()
        self.layer_0_d.reset_parameters()
        self.layer_1_d.reset_parameters()
        self.layer_2_d.reset_parameters()
        self.layer_3_d.reset_parameters()

    def forward(self, x, edge_index, node_counts, pool_maps, sub_edges):
        x_0 = self.layer_0_e(x, edge_index)
        x_0 = self.pool(x_0, node_counts[0], pool_maps[0])

        x_1 = self.layer_1_e(x_0, sub_edges[0])
        x_1 = self.pool(x_1, node_counts[1], pool_maps[1])

        x_2 = self.layer_2_e(x_1, sub_edges[1])
        x_2 = self.pool(x_2, node_counts[2], pool_maps[2])

        x_3 = self.layer_3_e(x_2, sub_edges[2])
        x_3 = self.pool(x_3, node_counts[3], pool_maps[3])

        x_4 = self.layer_4(x_3, sub_edges[3])
        x_4 = self.unpool(x_4, pool_maps[3])

        x_3 = self.layer_3_d(x_4, sub_edges[2])
        x_3 = self.unpool(x_3, pool_maps[2])

        x_2 = self.layer_2_d(x_3, sub_edges[1])
        x_2 = self.unpool(x_2, pool_maps[1])

        x_1 = self.layer_1_d(x_2, sub_edges[0])
        x_1 = self.unpool(x_1, pool_maps[0])

        x_0 = self.layer_0_d(x_1, edge_index)
        x_out = self.layer_out(x_0)

        return self.tanh(x_out) * 0.5

    @staticmethod
    def pool(x, node_count, pool_map):
        x_pooled = torch.zeros((node_count, x.shape[1]), dtype=x.dtype).to(x.device)
        torch_scatter.scatter_max(x, pool_map, dim=0, out=x_pooled)
        return x_pooled

    @staticmethod
    def unpool(x, pool_map):
        x_unpooled = x[pool_map, :]
        return x_unpooled


class GCNNet(nn.Module):

    def __init__(self, in_channels, out_channels, nf, dropout):
        super(GCNNet, self).__init__()

        self.gat_head = GCN(in_channels, nf, 5, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=InstanceNorm(nf))
        self.gat_tail = GCN(nf * 3, nf, 2, out_channels, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=InstanceNorm(nf))
        self.skip = GCN(in_channels, nf, 1, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=InstanceNorm(nf))
        self.global_head = GCN(in_channels, nf, 2, nf, dropout, torch.nn.LeakyReLU(0.02), aggr='max', norm=InstanceNorm(nf))
        self.tanh = torch.nn.Tanh()

    def reset_parameters(self):
        self.gat.reset_parameters()

    def forward(self, x, edge_index):
        x_head = self.gat_head(x, edge_index)
        x_skip = self.skip(x, edge_index)
        x_global = self.global_head(x, edge_index).mean(dim=0).unsqueeze(0).expand((x_head.shape[0], -1))
        x_out = self.gat_tail(torch.cat([x_head, x_skip, x_global], dim=1), edge_index)
        return self.tanh(x_out) * 0.5


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
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

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

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
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

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index.cpu(), edge_weight.cpu(), edge_index.cpu(),
                                         edge_weight.cpu(), num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index.cuda(), edge_weight.cuda())
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


if __name__ == "__main__":
    print(GATNet(7, 3, 24, 4, 8, 1, 0))
