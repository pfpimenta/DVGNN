# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Conv2d


def get_B_matrix(tem_size: int = 60, time_window_size: int = 12):
    """Matrix used for putting extremmely negative values
    in places we want the softmax to ignore it.
    The effect is that the softmax is applied independently in each
    time window of size 'time_window_size'.
    """
    A = np.zeros((tem_size, tem_size))
    for i in range(time_window_size):
        for j in range(time_window_size):
            A[i, j] = 1
            A[i + time_window_size, j + time_window_size] = 1
            A[i + time_window_size * 2, j + time_window_size * 2] = 1
    for i in range(time_window_size * 2):
        for j in range(time_window_size * 2):
            A[i + time_window_size * 3, j + time_window_size * 3] = 1
    B = (-1e13) * (1 - A)
    # B=(torch.tensor(B)).type(torch.float32).cuda()
    B = (torch.tensor(B)).type(torch.float32)
    return B


# temporal attention layer
class TATT_1(nn.Module):
    def __init__(self, c_in: int, num_nodes: int, tem_size: int):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)
        self.B_matrix = get_B_matrix()

    def forward(self, seq: torch.Tensor):
        # try:
        # DEBUG seq shape w T_cheby_conv_ds: torch.Size([16, 60, 64, 60])
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n

        # fix case where there is only 1 instance in batch
        if len(f1.shape) == 2:
            f1 = torch.unsqueeze(f1, dim=0)
        if len(f2.shape) == 2:
            f2 = torch.unsqueeze(f2, dim=0)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        device = next(self.parameters()).device
        self.B_matrix = self.B_matrix.to(device)
        coefs = torch.softmax(logits + self.B_matrix, -1)
        return coefs
