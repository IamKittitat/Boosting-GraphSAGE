import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.graphsage.encoders import Encoder
from src.graphsage.aggregators import MeanAggregator
from src.graphsage.model import SupervisedGraphSage

# CONSTANT
# features name -> Ex. GDMicro_T2D
# MEASUREMENT 
# PERC_VAL

def main():
    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, "data/2_OTU/GDMicro_T2D_features.csv"), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, "data/2_OTU/GDMicro_T2D_labels.csv"), header=None)
    labels = labels.to_numpy().flatten()

    # Calculate Distance matrix
    distance_matrix = cal_distance_matrix(features, "EUCLIDEAN")
    np.savetxt(os.path.join(CURRENT_DIR, "data/3_distance_matrix/GDMicro_T2D.csv"), distance_matrix, delimiter=",")

    # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
    distance_threshold = cal_distance_threshold(distance_matrix)
    tau_sick, tau_healthy = cal_neighbor_threshold(labels, perc_val=20, is_balanced=False)

    # Get Graph (Adjacency Matrix) from the distance matrix and threshold
    adj_matrix = md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels)
    np.savetxt(os.path.join(CURRENT_DIR, "data/4_adj_matrix/GDMicro_T2D.csv"), adj_matrix, delimiter=",", fmt="%d")

    # Preparing train/val/test | or stratify k-fold cv (k = 10)
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
    print("Validation F1:", f1_score(labels[val], val_output.detach().numpy().argmax(axis=1)))
    print("Average batch time:", np.mean(times))

        # Train model -> (feat_vec, adh, label) -> model + result?
        # Evaluate model
        # Log results
        # Save model
    return


if __name__ == "__main__":
    main()