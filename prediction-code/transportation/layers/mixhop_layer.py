# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn_layer import gcn_message_passing


class MixHopLayer(nn.Module):
    """MixHop GNN layer from the paper titled
    "MixHop: Higher-Order Graph Convolutional Architectures
    via Sparsified Neighborhood Mixing" by Abu-El-Haija et al. (2019)
    https://arxiv.org/abs/1905.00067
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        adjacency_powers: List[int],
        device: torch.device,
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
        self.device = device
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

        if self.weights[a_power].device != self.device:
            self.weights[a_power] = self.weights[a_power].to(self.device)

        # multiply features by weights
        x = torch.einsum("abcd, bx -> axcd", x, self.weights[a_power])

        # add bias
        if self.biases is not None:
            # optimized for GPU
            bias = (
                self.biases[a_power]
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .to(self.device)
            )
            x += bias

        # message passing
        if a_power > 0:
            x = gcn_message_passing(adj=adj, x=x, device=self.device, a_power=a_power)

        # activation function
        x = F.leaky_relu(x)

        return x

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Applies the MixHop operation:
        X{i+1} = concat([activation_func(Adj * X{i}{j} * W{i}{j} for j in adjacency_powers])
        where:
            x.shape : [batch_num, in_channels, num_nodes, tem_size]
            adj.shape : [batch_num, num_nodes, num_nodes]
            self.weights[i].shape : [in_channels, out_channels]
            self.biases[i].shape : [out_channels]
            x_out.shape : [batch_num, out_channels*len(adjacency_powers), num_nodes, tem_size]
        """
        x_out_list = []
        for a_power in self.adjacency_powers:
            x_out = self.gcn_conv(x=x, adj=adj, a_power=a_power)
            x_out_list.append(x_out)
        x_out = torch.concat(x_out_list, dim=1)
        return x_out
