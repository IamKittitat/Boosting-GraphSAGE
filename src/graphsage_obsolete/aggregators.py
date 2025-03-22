import torch
import torch.nn as nn
import random

class MeanAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, all_neighbors, num_sample):
        if num_sample is not None:
            samp_neighs = [list(random.sample(list(neigh), num_sample)) if len(neigh) >= num_sample else neigh for neigh in all_neighbors]
        else:
            samp_neighs = all_neighbors

        if self.gcn:
            samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]
        
        unique_nodes_list = list(set().union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        row_indices = [i for i in range(len(samp_neighs)) for _ in samp_neighs[i]]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if isinstance(self.features, nn.Embedding):
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        else:
            embed_matrix = self.features(unique_nodes_list, samp_neighs)
        
        print("AGG", mask.shape, embed_matrix.shape)
        to_feats = mask.mm(embed_matrix)
        return to_feats
