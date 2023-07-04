# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn_layer import GCNLayer
from layers.mixhop_layer import MixHopLayer
from layers.t_cheby_conv_ds import T_cheby_conv_ds
from layers.ttat import TATT_1
from torch.nn import Conv2d, LayerNorm


class ST_BLOCK_2(nn.Module):
    """Spatio-temporal block containing temporal convolutions, GNNs,
    and temporal attention layers"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_nodes: int,
        tem_size: int,
        K: int,
        Kt: int,
        device: torch.device,
        use_mixhop: bool = False,
    ):
        super(ST_BLOCK_2, self).__init__()
        # before: Chebychev conv
        # self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        # self.dynamic_gcn2 = T_cheby_conv_ds(c_out,c_out,K,Kt)
        # modification: MixHop layers or Vanilla GCN layers
        self.use_mixhop = use_mixhop
        if self.use_mixhop:
            # print("Using MixHop")
            adjacency_powers = [0, 1, 2]
            self.c_out = c_out * len(adjacency_powers)  # only for MixHop
            self.dynamic_gcn = MixHopLayer(
                input_size=c_out * len(adjacency_powers),
                output_size=c_out * 2,
                adjacency_powers=adjacency_powers,
                device=device,
            )
            self.dynamic_gcn2 = MixHopLayer(
                input_size=c_out,
                output_size=c_out,
                adjacency_powers=adjacency_powers,
                device=device,
            )
        else:
            # print("Using Vanilla GCN")
            self.c_out = c_out
            self.dynamic_gcn = GCNLayer(
                input_size=c_out, output_size=c_out * 2, device=device
            )
            self.dynamic_gcn2 = GCNLayer(
                input_size=c_out, output_size=c_out, device=device
            )

        self.K = K
        self.tem_size = tem_size
        self.time_conv = Conv2d(
            c_in, c_out, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True
        )
        self.bn = LayerNorm([self.c_out, num_nodes, tem_size])
        self.TATT_1 = TATT_1(self.c_out, num_nodes, tem_size)

    def forward(self, x: torch.Tensor, supports: torch.Tensor, adj_r: torch.Tensor):
        shape = supports.shape

        x_input1 = self.time_conv(x)  # "Feature sampling" from the article

        if shape[0] == 207:
            x_input1 = F.leaky_relu(x_input1)
        if shape[0] == 170:
            x_input1 = F.leaky_relu(x_input1)

        x_input2 = self.dynamic_gcn2(x_input1, adj_r)
        x_1 = self.dynamic_gcn(x_input2, adj_r)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        # x_1=F.dropout(x_1,0.5,self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum("bcnl,blq->bcnq", x_1, T_coef)
        if self.use_mixhop:
            # MixHop
            out = self.bn(
                F.leaky_relu(x_1) + torch.concat([x_input1, x_input1, x_input1], dim=1)
            )
        else:
            # Vanilla GCN
            out = self.bn(F.leaky_relu(x_1) + x_input1)
        return out, adj_r, T_coef


""" old version:
    def forward(self, x: torch.Tensor, supports: torch.Tensor, adj_r: torch.Tensor):
        # breakpoint()
        shape = supports.shape

        x_input1 = self.time_conv(x)

        if shape[0] == 207:
            x_input1 = F.leaky_relu(x_input1)
        if shape[0] == 170:
            x_input1 = F.leaky_relu(x_input1)

        x_input2 = self.dynamic_gcn2(x_input1, adj_r)
        x_1 = self.dynamic_gcn(x_input2, adj_r)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        # x_1=F.dropout(x_1,0.5,self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum("bcnl,blq->bcnq", x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input1)
        return out, adj_r, T_coef
"""
