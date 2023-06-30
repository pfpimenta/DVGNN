# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn_layer import normalize_adjacency


class MixHopLayer(nn.Module):
    """MixHop GNN layer from the paper titled
    "MixHop: Higher-Order Graph Convolutional Architectures
    via Sparsified Neighborhood Mixing" by Abu-El-Haija et al. (2019)
    https://arxiv.org/abs/1905.00067
    """

    # def __init__(self, input_size: int, output_size: int, adjacency_powers: List[int], device: torch.device, use_bias: bool =True):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        adjacency_powers: List[int],
        use_bias: bool = True,
    ):
        """
        input_size: size of the input features
        output_size: size of the output features for each adjacency power.
            The output size of the layer will be == len(adjacency_powers)*output_size.
        adjacency_powers: list of integers containing powers of adjacency matrix.
        """
        super(MixHopLayer, self).__init__()
        self.adjacency_powers = adjacency_powers
        self.weights = {}
        if use_bias:
            self.biases = {}
        else:
            self.register_parameter("biases", None)

        for a_power in adjacency_powers:
            self.weights[a_power] = nn.Parameter(
                torch.FloatTensor(torch.zeros(size=(input_size, output_size)))
            )
            if use_bias:
                self.biases[a_power] = nn.Parameter(
                    torch.FloatTensor(torch.zeros(size=(output_size,)))
                )

        self.initialize_weights()

    def initialize_weights(self):
        for a_power in self.adjacency_powers:
            nn.init.xavier_uniform_(self.weights[a_power])
            if self.biases is not None:
                nn.init.zeros_(self.biases[a_power])

    def gcn_conv(
        self, x: torch.Tensor, adj: torch.Tensor, a_power: int
    ) -> torch.Tensor:
        batch_size, num_features, num_nodes, time_size = x.shape
        output_features_size = self.weights[a_power].shape[1]

        # x = x @ self.weight
        # x_out = torch.zeros(size=(batch_size, output_features_size, num_nodes, time_size))
        # for batch_i in range(batch_size):
        #     for time_i in range(time_size):
        #         x_out[batch_i, :, :, time_i] = (x[batch_i, :, :, time_i].clone().transpose(0, 1) @ self.weights[a_power]).transpose(0, 1)
        try:
            x = torch.einsum("abcd, bx -> axcd", x, self.weights[a_power])
        except Exception as e:
            print(e)
            breakpoint()

        # x += self.bias
        if self.biases is not None:
            for batch_i in range(batch_size):
                for time_i in range(time_size):
                    x[batch_i, :, :, time_i] = (
                        x[batch_i, :, :, time_i].clone().transpose(0, 1)
                        + self.biases[a_power]
                    ).transpose(0, 1)

        # get simmetrically normalized adjacency matrix from Kipf & Welling (2017)
        # x = torch.sparse.mm(adj, x)
        if a_power == 0:
            # TODO verify this
            pass
            # adj = torch.eye(adj.shape[1])
            # for batch_i in range(batch_size):
            #     for time_i in range(time_size):
            #         x_out[batch_i, :, :, time_i] = torch.sparse.mm(adj[batch_i, :, :], (x_out[batch_i, :, :, time_i].clone()).transpose(0,1)).transpose(0,1)
        else:
            # device = next(self.parameters()).device
            # adj = normalize_adjacency(adj=adj, device=device)
            for batch_i in range(batch_size):
                for time_i in range(time_size):
                    for _ in range(a_power):
                        x[batch_i, :, :, time_i] = torch.sparse.mm(
                            adj[batch_i, :, :],
                            (x[batch_i, :, :, time_i].clone()).transpose(0, 1),
                        ).transpose(0, 1)

        x = F.leaky_relu(x)
        return x

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # H{i+1} = concat(activation_func(Adj * H{i}{j} * W{i}{j}) for j in adjacency_powers
        x_out_list = []
        for a_power in self.adjacency_powers:
            x_out = self.gcn_conv(x=x, adj=adj, a_power=a_power)
            x_out_list.append(x_out)
        x_out = torch.concat(x_out_list, dim=1)
        return x_out
        # x.shape : [batch_num,in_channels,num_nodes,tem_size]
        # adj.shape : [num_nodes, num_nodes]
        # self.weight.shape : torch.Size([input_size, output_size])
        # self.bias.shape : torch.Size([output_size])
        # x_out.shape

        """
        breakpoint()
        # TODO
        # *** RuntimeError: mat1 and mat2 shapes cannot be multiplied (65536x60 and 64x128)
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj, x)
        """
