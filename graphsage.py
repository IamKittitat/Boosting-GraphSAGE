import torch
import torch.nn as nn
import numpy as np
import time
import random
from sklearn.metrics import f1_score

from src.graphsage.encoders import Encoder
from src.graphsage.aggregators import MeanAggregator
from src.graphsage.model import SupervisedGraphSage

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if info[-1] not in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    num_nodes = len(node_map)  # Number of nodes in the graph
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # Create a matrix of zeros

    with open("cora/cora.cites") as fp:
        for line in fp:
            info = line.strip().split()
            paper1 = node_map[info[0]] 
            paper2 = node_map[info[1]]
            adj_matrix[paper1, paper2] = 1
            adj_matrix[paper2, paper1] = 1
        return feat_data, labels, adj_matrix

def run_cora():
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists = load_cora()
    num_nodes = len(feat_data)
    num_feats = len(feat_data[0])
    features = nn.Embedding(num_nodes, num_feats)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, num_feats, 128, adj_lists, agg1, num_sample=5, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2, num_sample=5, base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test, val, train = rand_indices[:1000], rand_indices[1000:1500], rand_indices[1500:]

    optimizer = torch.optim.SGD(graphsage.parameters(), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, torch.LongTensor(labels[np.array(batch_nodes)]))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.detach().numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
