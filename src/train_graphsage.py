import numpy as np
import torch
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


from src.graphsage.model import SupervisedGraphSage

def train_graphsage(features, adj_matrix, labels):
    graphsage = SupervisedGraphSage(features, adj_matrix, 2, 5)
    train, val_test = train_test_split(np.arange(len(features)), test_size=0.3, random_state=42)
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