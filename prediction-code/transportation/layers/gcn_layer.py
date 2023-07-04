# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO move to another file
def gcn_message_passing(
    adj: torch.Tensor, x: torch.Tensor, device: torch.device, a_power: int = 1
):
    batch_size, num_features, num_nodes, time_size = x.shape

    # get simmetrically normalized adjacency matrix from Kipf & Welling (2017)
    adj = normalize_adjacency(adj=adj, device=device)
    adj_power = adj.pow(a_power)
    adj_power = adj_power.to_dense()  # Convert sparse matrix to dense

    # Reshape adj_power to match the dimensions of x
    adj_power = adj_power.unsqueeze(1).repeat(
        1, time_size, 1, 1
    )  # Shape: [batch_size, time_size, n, n]

    # Reshape x to match the dimensions of adj_power
    x_new = x.permute(
        0, 3, 2, 1
    ).contiguous()  # Shape: [batch_size, time_size, num_nodes, num_features]
    x_new = x_new.view(
        batch_size * time_size, num_nodes, num_features
    )  # Reshape to [batch_size*time_size, num_nodes, num_features]

    # Perform batch matrix multiplication
    result = torch.bmm(
        adj_power.view(-1, num_nodes, num_nodes), x_new
    )  # Shape: [batch_size*time_size, num_nodes, num_features]

    # Reshape the result back to the original shape of x
    result = result.view(batch_size, time_size, num_nodes, num_features)
    result = result.permute(
        0, 3, 2, 1
    )  # Shape: [batch_size, num_features, num_nodes, time_size]

    # Assign the result back to x
    x_new = result.clone()
    return x_new


# TODO move to another file
def normalize_adjacency(adj: torch.Tensor, device: torch.device):
    """Normalizes the adjacency matrix according to the
    paper by Kipf et al.
    https://arxiv.org/pdf/1609.02907.pdf
    adj.shape: [num_batch, num_nodes, num_nodes]
    """
    adj = adj + torch.eye(adj.shape[1]).to(device)

    # TODO problem? adj matrix is not only 0s and 1s... verify
    # binary_adj = (adj != 0).to(torch.int32)
    node_degrees = torch.sum(adj, dim=1)
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

    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device,
        use_bias: bool = True,
    ):
        super(GCNLayer, self).__init__()
        self.device = device
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

    def gcn_conv(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, num_nodes, time_size = x.shape
        output_features_size = self.weight.shape[1]

        if self.weight.device != self.device:
            self.weight = self.weight.to(self.device)

        # originally: x = x @ self.weight
        x = torch.einsum("abcd, bx -> axcd", x, self.weight)

        # originally: x += self.bias
        if self.bias is not None:
            # optimized for GPU
            bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
            x += bias

        # originally: x = torch.sparse.mm(adj, x)
        x = gcn_message_passing(a_power=1, adj=adj, x=x, device=self.device)

        x = F.leaky_relu(x)
        return x

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Applies the following operation:
        X{i+1} = activation_func(Adj * X{i} * W{i})
        where:
            x.shape : [batch_num, in_channels, num_nodes, tem_size]
            adj.shape : [num_nodes, num_nodes]
            self.weight.shape : [in_channels, out_channels]
            self.bias.shape : [out_channels]
            x_out.shape : [batch_num, out_channels, num_nodes, tem_size]
        """
        x = self.gcn_conv(x=x, adj=adj)
        return x
