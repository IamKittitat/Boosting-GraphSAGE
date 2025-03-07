import torch
import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.graphsage.model import SupervisedGraphSage

def train_graphsage(features, adj_matrix, labels, config):
    graphsage = SupervisedGraphSage(features=features, 
                                    adj_matrix=adj_matrix, 
                                    num_classes=config['model']['num_classes'], 
                                    num_layers=config['model']['num_layers'], 
                                    num_sample1=config['model']['num_sample1'], 
                                    num_sample2=config['model']['num_sample2'], 
                                    embed_dim=config['model']['embed_dim']
                                    )
    train, val_test = train_test_split(np.arange(len(features)), test_size=config['data']['val_test_ratio'], stratify=labels, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, stratify=labels[val_test], random_state=42)

    optimizer = torch.optim.Adam(graphsage.parameters(), lr=config['model']['lr'])

    for epoch in range(config['model']['epoch']):
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(train, torch.LongTensor(labels[np.array(train)]))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{config['model']['epoch']}, Loss: {loss.item()}")

    val_output = graphsage(val).detach().numpy()
    test_output = graphsage(test).detach().numpy()

    val_f1 = f1_score(labels[val], val_output.argmax(axis=1))
    test_f1 = f1_score(labels[test], test_output.argmax(axis=1))
    print("Validation F1:", val_f1)
    print("Test F1:", test_f1)
    return val_f1, test_f1
