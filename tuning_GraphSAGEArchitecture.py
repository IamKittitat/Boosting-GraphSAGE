import torch
import random
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import optuna
from src.graphsage.model import SupervisedGraphSage
from src.load_config import load_config


def objective(trial):
    config = load_config()
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()
    adj_matrix = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['adj_matrix_output']), header=None)
    adj_matrix = adj_matrix.to_numpy()

    num_sample1 = trial.suggest_int('num_sample1', 10, 100, step=5)
    num_sample2 = trial.suggest_int('num_sample2', 10, 100, step=5)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    
    graphsage = SupervisedGraphSage(features=features, 
                                    adj_matrix=adj_matrix, 
                                    num_classes=config['model']['num_classes'], 
                                    num_layers=num_layers, 
                                    num_sample1=num_sample1, 
                                    num_sample2=num_sample2, 
                                    embed_dim=config['model']['embed_dim'])

    train, val_test = train_test_split(np.arange(len(features)), test_size=config['data']['val_test_ratio'], stratify=labels, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, stratify=labels[val_test], random_state=42)

    optimizer = torch.optim.Adam(graphsage.parameters(), lr=config['model']['lr'])

    for epoch in range(config['model']['epoch']):
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(train, torch.LongTensor(labels[np.array(train)]))
        loss.backward()
        optimizer.step()

    val_output = graphsage(val)
    test_output = graphsage(test)

    val_output = val_output.detach().numpy()
    test_output = test_output.detach().numpy()

    val_f1 = f1_score(labels[val], val_output.argmax(axis=1), average='macro')
    test_f1 = f1_score(labels[test], test_output.argmax(axis=1), average='macro')

    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation F1 score: {study.best_value}")
