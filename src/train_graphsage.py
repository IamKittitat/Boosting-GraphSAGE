import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.graphsage.encoders import Encoder
from src.graphsage.aggregators import MeanAggregator
from src.graphsage.model import SupervisedGraphSage

def train_graphsage(features, adj_matrix, labels):
    num_nodes = len(features)
    num_feats = len(features[0])
    features_nn = nn.Embedding(num_nodes, num_feats)
    features_nn.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
    agg1 = MeanAggregator(features_nn, cuda=False)
    enc1 = Encoder(features_nn, num_feats, 128, adj_matrix, agg1, num_sample=5, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_matrix, agg2, num_sample=5, base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(2, enc2)
    train, val_test = train_test_split(np.arange(num_nodes), test_size=0.3, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)

    optimizer = torch.optim.SGD(graphsage.parameters(), lr=0.7)
    for batch in range(100):
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(train, torch.LongTensor(labels[np.array(train)]))
        loss.backward()
        optimizer.step()
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.detach().numpy().argmax(axis=1)))