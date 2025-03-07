import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_matrix, aggregator, num_sample=10, base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_matrix = adj_matrix 
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

        self.weight = nn.Parameter(torch.empty(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        neighbors = [set(np.where(self.adj_matrix[node] == 1)[0]) for node in nodes]
        neigh_feats = self.aggregator.forward(nodes, neighbors, self.num_sample)

        if not self.gcn:
            self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.t()))
        return combined
