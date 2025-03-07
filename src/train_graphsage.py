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


def train_boosting_graphsage(features, adj_matrix, labels, config):
    M = config['model']['base_estimators']    
    base_models = []
    model_weights = []
    
    train, val_test = train_test_split(np.arange(len(features)), test_size=config['data']['val_test_ratio'], stratify=labels, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, stratify=labels[val_test], random_state=42)
    
    N = len(train)
    weights = np.ones(N) / N 
    for m in range(M):
        print(f"Base estimators: {m+1}/{M}")
        bootstrap_indices = np.random.choice(train, size=len(train), replace=True, p=weights)
        bootstrap_features = features[bootstrap_indices]
        bootstrap_adj = adj_matrix[bootstrap_indices][:, bootstrap_indices]
        bootstrap_labels = labels[bootstrap_indices]

        model = SupervisedGraphSage(features=bootstrap_features, adj_matrix=bootstrap_adj,
                                    num_classes=config['model']['num_classes'],
                                    num_layers=config['model']['num_layers'],
                                    num_sample1=config['model']['num_sample1'],
                                    num_sample2=config['model']['num_sample2'],
                                    embed_dim=config['model']['embed_dim'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
        
        for epoch in range(config['model']['epoch']):
            optimizer.zero_grad()
            loss = model.loss(np.arange(N), torch.LongTensor(bootstrap_labels))
            loss.backward()
            optimizer.step()
        
        train_preds = model(np.arange(N)).detach().numpy().argmax(axis=1)
        incorrect = (train_preds != bootstrap_labels)
        error_m = np.sum(weights * incorrect) / np.sum(weights)
        # If error is too high, discard the model
        if error_m > 0.5:
            print(error_m)
            continue
        
        alpha_m = 0.5 * np.log((1 - error_m) / (error_m + 1e-10))
        base_models.append(model)
        model_weights.append(alpha_m)

        weights *= np.exp(alpha_m * incorrect)
        weights /= np.sum(weights)
    
    def boosting_predict(test_indices):
        final_predictions = np.zeros((len(test_indices), config['model']['num_classes']))
        for model, weight in zip(base_models, model_weights):
            predictions = model(test_indices).detach().numpy()
            final_predictions += weight * predictions
        return final_predictions.argmax(axis=1)
    
    print(len(labels))
    print(val)
    val_f1 = f1_score(labels[val], boosting_predict(val))
    test_f1 = f1_score(labels[test], boosting_predict(test))
    
    print("Validation F1:", val_f1)
    print("Test F1:", test_f1)
    return val_f1, test_f1
