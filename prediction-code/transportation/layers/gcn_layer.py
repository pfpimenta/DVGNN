# -*- coding: utf-8 -*-
# taken from https://github.com/senadkurtisi/pytorch-GCN/blob/main/src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO move to another file
def normalize_adjacency(adj: torch.Tensor, device: torch.device):
    """Normalizes the adjacency matrix according to the
    paper by Kipf et al.
    https://arxiv.org/pdf/1609.02907.pdf
    adj.shape: [num_batch, num_nodes, num_nodes]
    """
    adj = adj + torch.eye(adj.shape[1]).to(device)

    # TODO problem? adj matrix is not only 0s and 1s
    binary_adj = (adj != 0).to(torch.int32)
    node_degrees = torch.sum(binary_adj, dim=1)
    node_degrees = torch.pow(node_degrees, -0.5)
    node_degrees[node_degrees == float("inf")] = 0.0
    node_degrees[node_degrees != node_degrees] = 0.0
    degree_matrix = torch.diag_embed(node_degrees).to(torch.float32)

    normalized_adj = degree_matrix @ adj @ degree_matrix
    return normalized_adj


class GCNLayer(nn.Module):
    """Vanilla GCN from Kipf & Welling (2017)
    "Semi-supervised classification with graph convolutional networks."
    https://arxiv.org/pdf/1609.02907.pdf
    (adapted from
    https://github.com/senadkurtisi/pytorch-GCN/blob/main/src/model.py)
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(torch.zeros(size=(input_size, output_size)))
        )
        if use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(torch.zeros(size=(output_size,)))
            )
        else:
            self.register_parameter("bias", None)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # H{i+1} = activation_func(Adj * H{i} * W{i})

        # x.shape : [batch_num,in_channels,num_nodes,tem_size]
        # adj.shape : [num_nodes, num_nodes]
        # self.weight.shape : torch.Size([input_size, output_size])
        # self.bias.shape : torch.Size([output_size])
        # x_out.shape
        batch_size, num_features, num_nodes, time_size = x.shape

        # TODO check if these for loops make sense
        # x = x @ self.weight
        # output_features_size = self.weight.shape[1]
        # x_out = torch.zeros(size=(batch_size, output_features_size, num_nodes, time_size))
        # for batch_i in range(batch_size):
        #     for time_i in range(time_size):
        #         x_out[batch_i, :, :, time_i] = (x[batch_i, :, :, time_i].clone().transpose(0, 1) @ self.weight).transpose(0, 1)
        # TODO use torch.einsum instead ?
        # x_out_ein = torch.zeros(size=(batch_size, output_features_size, num_nodes, time_size))
        x = torch.einsum("abcd, bx -> axcd", x, self.weight)

        # x += self.bias
        if self.bias is not None:
            for batch_i in range(batch_size):
                for time_i in range(time_size):
                    x[batch_i, :, :, time_i] = (
                        x[batch_i, :, :, time_i].clone().transpose(0, 1) + self.bias
                    ).transpose(0, 1)
        # TODO remove for somehow
        # x_add = torch.add(x, self.bias)
        # breakpoint()

        # get simmetrically normalized adjacency matrix from Kipf & Welling (2017)
        # x = torch.sparse.mm(adj, x)
        device = next(self.parameters()).device
        adj = normalize_adjacency(adj=adj, device=device)
        for batch_i in range(batch_size):
            for time_i in range(time_size):
                x[batch_i, :, :, time_i] = torch.sparse.mm(
                    adj[batch_i, :, :],
                    (x[batch_i, :, :, time_i].clone()).transpose(0, 1),
                ).transpose(0, 1)

        x = F.leaky_relu(x)

        return x
