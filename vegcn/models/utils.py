#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl
import dgl.nn as dgl_nn


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        if features.dim() == 2:
            x = torch.spmm(A, features)
        elif features.dim() == 3:
            x = torch.bmm(A, features)
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg, dropout=0):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gcn_layer = dgl_nn.conv.GraphConv(in_dim, out_dim, bias=True)
        self.gcn_layer.reset_parameters()
        self.dropout = dropout

    def forward(self, dgl_g, features):
        feat_dim = features.shape[-1]
        assert (feat_dim == self.in_dim)
        out = self.gcn_layer(dgl_g, features)
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)
        return out
