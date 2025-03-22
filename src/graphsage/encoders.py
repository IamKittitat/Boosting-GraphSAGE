import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, aggregator, num_sample, base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

        self.weight = nn.Parameter(torch.empty(embed_dim, 2*self.feat_dim if self.gcn else self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, all_neighbors):
        neigh_feats = self.aggregator.forward(nodes, all_neighbors, self.num_sample)
        if self.gcn:
            self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        x = combined
        combined = F.relu(self.weight.mm(combined.t()))
        print("ENCODER", self.weight.shape, x.t().shape, combined.shape)
        return combined
