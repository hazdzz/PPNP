import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

# PPNP or APPNP
class PPNP(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(PPNP, self).__init__()
        self.graph_aug_linears = nn.ModuleList()
        self.K = K
        self.graph_aug_linears.append(layers.BackwardLinear(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        for k in range(1, K-1):
            self.graph_aug_linears.append(layers.BackwardLinear(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        self.graph_aug_linears.append(layers.BackwardLinear(in_features=n_hid, out_features=n_class, bias=enable_bias))
        self.relu = nn.ReLU()
        self.dropout = layers.MixedDropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, filter):
        x = self.dropout(x)
        for k in range(self.K-1):
            x = self.graph_aug_linears[k](x)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.graph_aug_linears[-1](x)
        if filter.is_sparse:
            x = torch.sparse.mm(filter, x)
        else:
            x = torch.mm(filter, x)
        x = self.log_softmax(x)

        return x