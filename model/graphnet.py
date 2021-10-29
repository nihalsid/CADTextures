import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.nn import init
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.nn import TopKPooling, GCNConv, BatchNorm, InstanceNorm, GraphNorm
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_geometric.utils.repeat import repeat
from torch_geometric.nn.models.basic_gnn import GraphSAGE, GCN, GIN, GAT
from torch_sparse import spspmm


def pool(x, node_count, pool_map, pool_op='max'):
    if pool_op == 'max':
        x_pooled = torch.ones((node_count, x.shape[1]), dtype=x.dtype).to(x.device) * (x.min().detach() - 1e-3)
        torch_scatter.scatter_max(x, pool_map, dim=0, out=x_pooled)
    elif pool_op == 'mean':
        x_pooled = torch.zeros((node_count, x.shape[1]), dtype=x.dtype).to(x.device)
        torch_scatter.scatter_mean(x, pool_map, dim=0, out=x_pooled)
    return x_pooled


def unpool(x, pool_map):
    x_unpooled = x[pool_map, :]
    return x_unpooled


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
        x_0 = pool(x_0, node_counts[0], pool_maps[0])

        x_1 = self.layer_1_e(x_0, sub_edges[0])
        x_1 = pool(x_1, node_counts[1], pool_maps[1])

        x_2 = self.layer_2_e(x_1, sub_edges[1])
        x_2 = pool(x_2, node_counts[2], pool_maps[2])

        x_3 = self.layer_3_e(x_2, sub_edges[2])
        x_3 = pool(x_3, node_counts[3], pool_maps[3])

        x_4 = self.layer_4(x_3, sub_edges[3])
        x_4 = unpool(x_4, pool_maps[3])

        x_3 = self.layer_3_d(x_4, sub_edges[2])
        x_3 = unpool(x_3, pool_maps[2])

        x_2 = self.layer_2_d(x_3, sub_edges[1])
        x_2 = unpool(x_2, pool_maps[1])

        x_1 = self.layer_1_d(x_2, sub_edges[0])
        x_1 = unpool(x_1, pool_maps[0])

        x_0 = self.layer_0_d(x_1, edge_index)
        x_out = self.layer_out(x_0)

        return self.tanh(x_out) * 0.5


@torch.jit.script
def swish(x):
    # swish
    return x * torch.sigmoid(x)


class GResNetBlock(nn.Module):

    def __init__(self, nf_in, nf_out, norm, activation, aggr):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = norm(nf_in)
        self.conv_0 = SAGEConv(nf_in, nf_out, aggr=aggr)
        self.norm_1 = norm(nf_out)
        self.conv_1 = SAGEConv(nf_out, nf_out, aggr=aggr)
        self.activation = activation
        if nf_in != nf_out:
            self.nin_shortcut = nn.Linear(nf_in, nf_out)

    def forward(self, x, edge_index):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h, edge_index)

        h = self.norm_1(h)
        h = self.activation(h)
        h = self.conv_1(h, edge_index)

        if self.nf_in != self.nf_out:
            x = self.nin_shortcut(x)

        return x + h


class GAttnBlock(nn.Module):

    def __init__(self, nf, norm):
        super().__init__()
        self.in_channels = nf
        self.norm = norm(nf)
        self.q = nn.Linear(nf, nf)
        self.k = nn.Linear(nf, nf)
        self.v = nn.Linear(nf, nf)
        self.proj_out = nn.Linear(nf, nf)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        N, c = q.shape

        q = q.unsqueeze(0)
        k = k.t().unsqueeze(0)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.t().unsqueeze(0)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.squeeze(0).permute((1, 0))
        h_ = self.proj_out(h_)

        return x + h_


class BigGraphSAGEEncoderDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, nf, aggr, num_pools=5):
        super().__init__()
        self.num_pools = num_pools
        norm = GraphNorm
        self.activation = nn.LeakyReLU(0.02)
        self.enc_conv_in = SAGEConv(in_channels, nf, aggr=aggr)

        self.down_0_block_0 = GResNetBlock(nf, nf, norm, self.activation, aggr)
        self.down_0_block_1 = GResNetBlock(nf, nf, norm, self.activation, aggr)

        self.down_1_block_0 = GResNetBlock(nf, nf, norm, self.activation, aggr)
        self.down_1_block_1 = GResNetBlock(nf, nf, norm, self.activation, aggr)

        self.down_2_block_0 = GResNetBlock(nf, nf * 2, norm, self.activation, aggr)
        self.down_2_block_1 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)

        self.down_3_block_0 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)
        self.down_3_block_1 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)

        self.down_4_block_0 = GResNetBlock(nf * 2, nf * 4, norm, self.activation, aggr)
        self.down_4_block_1 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        # self.down_4_attn_block_0 = GAttnBlock(nf * 4, norm)
        # self.down_4_attn_block_1 = GAttnBlock(nf * 4, norm)

        self.enc_mid_block_0 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        # self.enc_mid_attn_0 = GAttnBlock(nf * 4, norm)
        self.enc_mid_block_1 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)

        self.enc_out_conv = SAGEConv(nf * 4, nf * 2, aggr=aggr)
        self.enc_out_norm = norm(nf * 4)

        self.dec_conv_in = SAGEConv(nf * 2, nf * 4, aggr=aggr)

        self.dec_mid_block_0 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        # self.dec_mid_attn_0 = GAttnBlock(nf * 4, norm)
        self.dec_mid_block_1 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)

        self.up_0_block_0 = GResNetBlock(nf, nf, norm, self.activation, aggr)
        self.up_0_block_1 = GResNetBlock(nf, nf, norm, self.activation, aggr)
        self.up_0_block_2 = GResNetBlock(nf, nf, norm, self.activation, aggr)

        self.up_1_block_0 = GResNetBlock(nf * 2, nf, norm, self.activation, aggr)
        self.up_1_block_1 = GResNetBlock(nf, nf, norm, self.activation, aggr)
        self.up_1_block_2 = GResNetBlock(nf, nf, norm, self.activation, aggr)

        self.up_2_block_0 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)
        self.up_2_block_1 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)
        self.up_2_block_2 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)

        self.up_3_block_0 = GResNetBlock(nf * 4, nf * 2, norm, self.activation, aggr)
        self.up_3_block_1 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)
        self.up_3_block_2 = GResNetBlock(nf * 2, nf * 2, norm, self.activation, aggr)

        self.up_4_block_0 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        self.up_4_block_1 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        self.up_4_block_2 = GResNetBlock(nf * 4, nf * 4, norm, self.activation, aggr)
        # self.up_4_attn_block_0 = GAttnBlock(nf * 4, norm)
        # self.up_4_attn_block_1 = GAttnBlock(nf * 4, norm)
        # self.up_4_attn_block_2 = GAttnBlock(nf * 4, norm)

        self.dec_out_norm = norm(nf)
        self.dec_out_conv = SAGEConv(nf, out_channels, aggr='max')
        self.tanh = nn.Tanh()

    def forward(self, x, graph_data):
        pool_ctr = 0
        x = self.enc_conv_in(x, graph_data.edge_index)
        x = self.down_0_block_0(x, graph_data.edge_index)
        x_0 = self.down_0_block_1(x, graph_data.edge_index)
        x = pool(x_0, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_1_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x_1 = self.down_1_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = pool(x_1, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_2_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x_2 = self.down_2_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        if self.num_pools == 5:
            x = pool(x_2, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
            pool_ctr += 1

        x = self.down_3_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x_3 = self.down_3_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = pool(x_3, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_4_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.down_4_attn_block_0(x)
        x_4 = self.down_4_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.down_4_attn_block_1(x)
        x = pool(x_4, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.enc_mid_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.enc_mid_attn_0(x)
        x = self.enc_mid_block_1(x, graph_data.sub_edges[pool_ctr - 1])

        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x, graph_data.sub_edges[pool_ctr - 1])

        x = self.dec_conv_in(x, graph_data.sub_edges[pool_ctr - 1])

        x = self.dec_mid_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.dec_mid_attn_0(x)
        x = self.dec_mid_block_1(x, graph_data.sub_edges[pool_ctr - 1])

        x = self.up_4_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.up_4_attn_block_0(x)
        x = self.up_4_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.up_4_attn_block_1(x)
        x = self.up_4_block_2(x, graph_data.sub_edges[pool_ctr - 1])
        # x = self.up_4_attn_block_2(x)
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1
        # x = torch.cat([x, x_3], dim=1)

        x = self.up_3_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_3_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_3_block_2(x, graph_data.sub_edges[pool_ctr - 1])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1

        # x = torch.cat([x, x_2], dim=1)

        x = self.up_2_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_2_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_2_block_2(x, graph_data.sub_edges[pool_ctr - 1])
        if self.num_pools == 5:
            x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
            pool_ctr -= 1

        # x = torch.cat([x, x_1], dim=1)

        x = self.up_1_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_1_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_1_block_2(x, graph_data.sub_edges[pool_ctr - 1])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1

        # x = torch.cat([x, x_0], dim=1)

        x = self.up_0_block_0(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_0_block_1(x, graph_data.sub_edges[pool_ctr - 1])
        x = self.up_0_block_2(x, graph_data.sub_edges[pool_ctr - 1])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1

        x = self.dec_out_norm(x)
        x = self.activation(x)
        x = self.dec_out_conv(x, graph_data.edge_index)

        return self.tanh(x) * 0.5


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


class FaceConv(nn.Module):

    @staticmethod
    def index(x, idx):
        return x[idx]

    @staticmethod
    def gather(x, idx):
        idx = idx[:, None].expand(idx.shape[0], x.shape[1])
        return x.gather(0, idx)

    @staticmethod
    def embed(x, idx):
        return torch.nn.functional.embedding(idx, x)

    def __init__(self, in_channels, out_channels, neighborhood_size):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.conv = nn.Conv2d(in_channels, out_channels, (1, neighborhood_size + 1), padding=0)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        f_ = [self.index(padded_x, face_neighborhood[:, i]) for i in range(self.neighborhood_size + 1)]
        conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3)
        return self.conv(conv_input).squeeze(-1).squeeze(-1)


class SpatialAttentionConv(nn.Module):

    @staticmethod
    def index(x, idx):
        return x[idx]

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_heads = max(1, in_channels // 32)
        self.attention = nn.MultiheadAttention(in_channels, self.num_heads, batch_first=True)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 2))

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        is_pad = face_is_pad[face_neighborhood[:, 1:]].unsqueeze(1).expand(-1, self.num_heads, -1).reshape(-1, 8).unsqueeze(1)
        f_ = torch.stack([self.index(padded_x, face_neighborhood[:, i]) for i in range(9)], dim=1)
        h = self.attention(f_[:, 0:1, :], f_[:, 1:, :], f_[:, 1:, :], attn_mask=is_pad, need_weights=False)[0]
        conv_input = torch.cat([f_[:, 0, :].unsqueeze(-1).unsqueeze(-1), h.squeeze(1).unsqueeze(-1).unsqueeze(-1)], dim=3)
        return self.conv(conv_input).squeeze(-1).squeeze(-1)


class Blur(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("weight", torch.ones((in_channels, 1, 1, 9)).float())
        self.register_buffer("blur_filter", torch.tensor([1/4, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16, 1/8, 1/16]).float())
        self.weight[:, :, 0, :] = self.blur_filter

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        f_ = [padded_x[face_neighborhood[:, i]] for i in range(9)]
        conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3)
        correction_factor = ((1 - face_is_pad[face_neighborhood].float()) * self.blur_filter.unsqueeze(0).expand(face_neighborhood.shape[0], -1)).sum(-1)
        correction_factor = correction_factor.unsqueeze(1).expand(-1, x.shape[1])
        return nn.functional.conv2d(conv_input, self.weight, groups=self.in_channels).squeeze(-1).squeeze(-1) / correction_factor


class WrappedLinear(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, _face_neighborhood, _is_pad, _pad_size):
        return self.linear(x)


class SymmetricFaceConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_0 = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_1 = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.weight_2 = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    @staticmethod
    def index(x, idx):
        return x[idx]

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.empty((self.out_channels, self.in_channels, 1, 9)))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def create_symmetric_conv_filter(self):
        return torch.cat([self.weight_0, self.weight_1, self.weight_2,
                          self.weight_1, self.weight_2, self.weight_1,
                          self.weight_2, self.weight_1, self.weight_2], dim=-1)

    def forward(self, x, face_neighborhood, face_is_pad, pad_size):
        padded_x = torch.zeros((pad_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded_x[torch.logical_not(face_is_pad), :] = x
        f_ = [self.index(padded_x, face_neighborhood[:, i]) for i in range(9)]
        conv_input = torch.cat([f.unsqueeze(-1).unsqueeze(-1) for f in f_], dim=3)
        return nn.functional.conv2d(conv_input, self.create_symmetric_conv_filter(), self.bias).squeeze(-1).squeeze(-1)


class FResNetBlock(nn.Module):

    def __init__(self, nf_in, nf_out, conv_layer, norm, activation):
        super().__init__()
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.norm_0 = norm(nf_in)
        self.conv_0 = conv_layer(nf_in, nf_out)
        self.norm_1 = norm(nf_out)
        self.conv_1 = conv_layer(nf_out, nf_out)
        self.activation = activation
        if nf_in != nf_out:
            self.nin_shortcut = conv_layer(nf_in, nf_out)

    def forward(self, x, face_neighborhood, face_is_pad, num_pad):
        h = x
        h = self.norm_0(h)
        h = self.activation(h)
        h = self.conv_0(h, face_neighborhood, face_is_pad, num_pad)

        h = self.norm_1(h)
        h = self.activation(h)
        h = self.conv_1(h, face_neighborhood, face_is_pad, num_pad)

        if self.nf_in != self.nf_out:
            x = self.nin_shortcut(x, face_neighborhood, face_is_pad, num_pad)

        return x + h


class BigFaceEncoderDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, nf, conv_layer, input_transform=None, num_pools=5):
        super().__init__()
        if input_transform is None:
            input_transform = conv_layer
        self.num_pools = num_pools
        norm = BatchNorm
        self.activation = nn.LeakyReLU()
        self.enc_conv_in = input_transform(in_channels, nf)
        self.down_0_block_0 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.down_0_block_1 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)

        self.down_1_block_0 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.down_1_block_1 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)

        self.down_2_block_0 = FResNetBlock(nf, nf * 2, conv_layer, norm, self.activation)
        self.down_2_block_1 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)

        self.down_3_block_0 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.down_3_block_1 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)

        self.down_4_block_0 = FResNetBlock(nf * 2, nf * 4, conv_layer, norm, self.activation)
        self.down_4_block_1 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        # self.down_4_attn_block_0 = GAttnBlock(nf * 4, conv_layer, norm)
        # self.down_4_attn_block_1 = GAttnBlock(nf * 4, conv_layer, norm)

        self.enc_mid_block_0 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        # self.enc_mid_attn_0 = GAttnBlock(nf * 4, conv_layer, norm)
        self.enc_mid_block_1 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)

        self.enc_out_conv = conv_layer(nf * 4, nf * 2)
        self.enc_out_norm = norm(nf * 4)

        self.dec_conv_in = conv_layer(nf * 2, nf * 4)

        self.dec_mid_block_0 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        # self.dec_mid_attn_0 = GAttnBlock(nf * 4, conv_layer, norm)
        self.dec_mid_block_1 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)

        self.up_0_block_0 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.up_0_block_1 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.up_0_block_2 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.blur_0 = Blur(nf)

        self.up_1_block_0 = FResNetBlock(nf * 2, nf, conv_layer, norm, self.activation)
        self.up_1_block_1 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.up_1_block_2 = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.blur_1 = Blur(nf)

        self.up_2_block_0 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.up_2_block_1 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.up_2_block_2 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.blur_2 = Blur(nf * 2)

        self.up_3_block_0 = FResNetBlock(nf * 4, nf * 2, conv_layer, norm, self.activation)
        self.up_3_block_1 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.up_3_block_2 = FResNetBlock(nf * 2, nf * 2, conv_layer, norm, self.activation)
        self.blur_3 = Blur(nf * 2)

        self.up_4_block_0 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        self.up_4_block_1 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        self.up_4_block_2 = FResNetBlock(nf * 4, nf * 4, conv_layer, norm, self.activation)
        self.blur_4 = Blur(nf * 4)
        # self.up_4_attn_block_0 = GAttnBlock(nf * 4, conv_layer, norm)
        # self.up_4_attn_block_1 = GAttnBlock(nf * 4, conv_layer, norm)
        # self.up_4_attn_block_2 = GAttnBlock(nf * 4, conv_layer, norm)

        self.dec_out_block = FResNetBlock(nf, nf, conv_layer, norm, self.activation)
        self.dec_out_norm = norm(nf)
        self.dec_out_linear = nn.Linear(nf, out_channels)

        self.tanh = nn.Tanh()

    def forward(self, x, graph_data):
        pool_ctr = 0
        x = self.enc_conv_in(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_0_block_0(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_0_block_1(x, graph_data['face_neighborhood'], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_1_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_1_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_2_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_2_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        if self.num_pools == 5:
            x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
            pool_ctr += 1

        x = self.down_3_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.down_3_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.down_4_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.down_4_attn_block_0(x)
        x = self.down_4_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.down_4_attn_block_1(x)
        x = pool(x, graph_data['node_counts'][pool_ctr], graph_data['pool_maps'][pool_ctr], pool_op='max')
        pool_ctr += 1

        x = self.enc_mid_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.enc_mid_attn_0(x)
        x = self.enc_mid_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.enc_out_norm(x)
        x = self.activation(x)
        x = self.enc_out_conv(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.dec_conv_in(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.dec_mid_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.dec_mid_attn_0(x)
        x = self.dec_mid_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.up_4_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.up_4_attn_block_0(x)
        x = self.up_4_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.up_4_attn_block_1(x)
        x = self.up_4_block_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        # x = self.up_4_attn_block_2(x)
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1
        x = self.blur_4(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.up_3_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_3_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_3_block_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1
        x = self.blur_3(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.up_2_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_2_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_2_block_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        if self.num_pools == 5:
            x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
            pool_ctr -= 1
            x = self.blur_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.up_1_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_1_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_1_block_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1
        x = self.blur_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])

        x = self.up_0_block_0(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_0_block_1(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = self.up_0_block_2(x, graph_data['sub_neighborhoods'][pool_ctr - 1], graph_data['is_pad'][pool_ctr], graph_data['pads'][pool_ctr])
        x = unpool(x, graph_data['pool_maps'][pool_ctr - 1])
        pool_ctr -= 1
        x = self.blur_0(x, graph_data['face_neighborhood'], graph_data['is_pad'][0], graph_data['pads'][0])

        x = self.dec_out_block(x, graph_data['face_neighborhood'], graph_data['is_pad'][0], graph_data['pads'][0])
        x = self.dec_out_norm(x)
        x = self.activation(x)
        x = self.dec_out_linear(x)

        return self.tanh(x) * 0.5
