#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl
import dgl.nn as dgl_nn

from dgl.base import DGLError
import dgl.function as fn


class u_mul_e_ele(nn.Module):
    '''
    Compute the input feature from neighbors
    '''
    def __init__(self):
        super(u_mul_e_ele, self).__init__()

    def forward(self, edges):
        return {'m': edges.src['h'] * edges.data['affine']}
    

# pylint: disable=W0235
class GraphConv_Concat(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConv_Concat, self).__init__()
        if norm not in ('none', 'both', 'left', 'right', 'affine'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats*2, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def direct_affine(self, edges):
        cos = nn.CosineSimilarity(dim=0)
        return {'m': cos(edges.src['h'], edges.dst['h']) * edges.src['h']}

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        ori_feat = feat

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
        
        if self._norm == 'left':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = norm * feat

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        # aggregate first then mult W
        graph.srcdata['h'] = feat
        
        '''
        graph.update_all(self.direct_affine,
                         fn.sum(msg='m', out='h'))
        '''
        
        graph.update_all(fn.u_mul_e('h', 'affine', 'm'),
                            fn.sum(msg='m', out='h'))
        
        
        '''
        graph.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))
        '''

        rst = torch.cat([ori_feat, graph.dstdata['h']], dim=-1)

        if weight is not None:
            rst = torch.matmul(rst, weight)

        if self._norm in ['both', 'right']:
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


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
        #self.gcn_layer = dgl_nn.conv.GraphConv(in_dim, out_dim, bias=True, norm='right')
        self.gcn_layer = GraphConv_Concat(in_dim, out_dim, norm='affine', bias=True)
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
