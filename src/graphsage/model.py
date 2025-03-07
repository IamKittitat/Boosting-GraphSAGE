import torch
import torch.nn as nn
from torch.nn import init
from src.graphsage.encoders import Encoder
from src.graphsage.aggregators import MeanAggregator

class SupervisedGraphSage(nn.Module):
    def __init__(self, features, adj_matrix, num_classes, num_sample1, num_sample2, embed_dim, num_layers=2):
        super(SupervisedGraphSage, self).__init__()
        self.xent = nn.CrossEntropyLoss()
        self.num_nodes = len(features)
        self.num_feats = len(features[0])
        self.features_nn = nn.Embedding(self.num_nodes, self.num_feats)
        self.features_nn.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)

        self.encoders = nn.ModuleList()
        self.aggregators = nn.ModuleList()

        agg = MeanAggregator(self.features_nn, cuda=False)
        enc = Encoder(self.features_nn, self.num_feats, embed_dim, adj_matrix, agg, num_sample=num_sample1, gcn=True, cuda=False)
        self.encoders.append(enc)
        self.aggregators.append(agg)

        prev_embed_fn = lambda nodes, prev_encoder=self.encoders[0]: prev_encoder(nodes).t()
        agg = MeanAggregator(prev_embed_fn, cuda=False)
        enc = Encoder(prev_embed_fn, self.encoders[0].embed_dim, embed_dim, adj_matrix, agg, num_sample=num_sample2, base_model=self.encoders[0], gcn=True, cuda=False)
        self.encoders.append(enc)
        self.aggregators.append(agg)
        
        for i in range(2, num_layers):
            prev_embed_fn = lambda nodes, prev_encoder=self.encoders[i-1]: prev_encoder(nodes).t()
            agg = MeanAggregator(prev_embed_fn, cuda=False)
            enc = Encoder(prev_embed_fn, self.encoders[i - 1].embed_dim, embed_dim, adj_matrix, agg, num_sample=num_sample2, base_model=self.encoders[i - 1], gcn=True, cuda=False)
            self.encoders.append(enc)
            self.aggregators.append(agg)

        self.weight = nn.Parameter(torch.empty(num_classes, self.encoders[-1].embed_dim))
        init.xavier_uniform_(self.weight)


    def forward(self, nodes):
        embeds = self.encoders[-1](nodes) 
        scores = self.weight.mm(embeds)
        return torch.sigmoid(scores.t())

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
