import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        # self.f1_weights = t.rand(2, 380)

    def forward(self, x, adj):
        g1 = self.gc1(x, adj)
        r1 = F.relu(g1)
        dr = F.dropout(r1, self.dropout, training=self.training)
        g2 = self.gc2(dr, adj)
        # r2 = F.relu(g2)
        # f1 = F.linear(r2, self.f1_weights)
        sm = F.log_softmax(g2, dim=1)
        # sm = F.softmax(g2, dim=1)

        return sm
