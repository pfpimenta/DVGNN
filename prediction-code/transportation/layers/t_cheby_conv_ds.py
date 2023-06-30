# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Conv2d


class T_cheby_conv_ds(nn.Module):
    """
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    """

    def __init__(self, c_in, c_out, K, Kt):
        super(T_cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(
            c_in_new,
            c_out,
            kernel_size=(1, Kt),
            padding=(0, 1),
            stride=(1, 1),
            bias=True,
        )
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape

        Ls = []
        L1 = adj
        # L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        L0 = torch.eye(nNode).repeat(nSample, 1, 1)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum("bcnl,bknq->bckql", x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out
