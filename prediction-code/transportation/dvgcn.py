# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
@author: Gorgen
@Fuction： （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”； #
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.st_block import ST_BLOCK_2
from torch.nn import BatchNorm2d, Conv2d, Parameter


class DVGCN(nn.Module):
    """Implementation of the Dynamic Diffusion-Variational Graph Neural Network model,
    from 'Dynamic Causal Explanation Based Diffusion-Variational Graph Neural
    Network for Spatio-temporal Forecasting', by Liang et al. (2023)
    https://arxiv.org/abs/2305.09703
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_nodes: int,
        week: int,
        day: int,
        recent: int,
        K: int,
        Kt: int,
        device: torch.device,
        use_mixhop: bool = False,
        adjacency_powers: List[int] = [0, 1, 2],
    ):
        """TODO param description"""
        super(DVGCN, self).__init__()

        if use_mixhop:
            print("Using MixHop layers")
            mixhop_coef = len(adjacency_powers)
        else:
            print("Using Vanilla GCN layers")
            mixhop_coef = 1

        self.device = device

        tem_size = week + day + recent
        self.block1 = ST_BLOCK_2(
            c_in=c_in,
            c_out=c_out,
            num_nodes=num_nodes,
            tem_size=tem_size,
            K=K,
            Kt=Kt,
            use_mixhop=use_mixhop,
            adjacency_powers=adjacency_powers,
            device=device,
        )
        self.block2 = ST_BLOCK_2(
            c_in=c_out * mixhop_coef,
            c_out=c_out,
            num_nodes=num_nodes,
            tem_size=tem_size,
            K=K,
            Kt=Kt,
            use_mixhop=use_mixhop,
            adjacency_powers=adjacency_powers,
            device=device,
        )
        self.bn = BatchNorm2d(c_in, affine=False)

        self.conv1 = Conv2d(
            c_out * mixhop_coef,
            1,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )
        self.conv2 = Conv2d(
            c_out * mixhop_coef,
            1,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )
        self.conv3 = Conv2d(
            c_out * mixhop_coef,
            1,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )
        self.conv4 = Conv2d(
            c_out * mixhop_coef,
            1,
            kernel_size=(1, 2),
            padding=(0, 0),
            stride=(1, 2),
            bias=True,
        )

        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(
        self,
        x_w: torch.Tensor,
        x_d: torch.Tensor,
        x_r: torch.Tensor,
        supports: torch.Tensor,
        adj_r: torch.Tensor,
    ):
        """TODO document what is each parameter
        x_w: node fearures week ?
        x_d: node fearures day ?
        x_r: node fearures recent ?
        supports: scaled_Laplacian(adj), where adj is the static adjacency matrix
        adj_r: Adjacency matrix recent ?
        """
        # print(f"x_w.shape: {x_w.shape}")
        # print(f"x_d.shape: {x_d.shape}")
        # print(f"x_r.shape: {x_r.shape}")
        # print(f"supports.shape: {supports.shape}")
        # print(f"adj_r.shape: {adj_r.shape}")

        # batch normalization
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r)
        # TODO concatenating features with different time scales?
        x = torch.cat((x_w, x_d, x_r), -1)

        # TODO what is A1 ?
        A = self.h + supports
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A1 = F.dropout(A, 0.8, self.training)

        # pass through 2 spatio-temporal blocks
        x, _, _ = self.block1(x, A1, adj_r)
        x, d_adj, t_adj = self.block2(x, A1, adj_r)

        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]
        x3 = x[:, :, :, 24:36]
        x4 = x[:, :, :, 36:60]

        # Residual layer
        x1 = self.conv1(x1).squeeze()
        x2 = self.conv2(x2).squeeze()
        x3 = self.conv3(x3).squeeze()
        x4 = self.conv4(x4).squeeze()  # b,n,l
        x = x1 + x2 + x3 + x4
        # output shapes:
        # x.shape: torch.Size([16, 64, 12])
        # d_adj.shape: torch.Size([16, 64, 64])
        # A.shape: torch.Size([64, 64])
        return x, d_adj, A
