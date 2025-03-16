import optuna
from src.load_config import load_config
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
from src.train_graphsage import train_boosting_graphsage

def objective(trial):
    config = load_config()
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()
    adj_matrix = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['adj_matrix_output']), header=None)
    adj_matrix = adj_matrix.to_numpy()

    M = trial.suggest_int('base_estimators', 5, 45, step=5) 
    config['model']['base_estimators'] = M

    val_f1, _ = train_boosting_graphsage(features, adj_matrix, labels, config)

    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=8) 

print(f"Best base_estimators: {study.best_params['base_estimators']}")
print(f"Best Validation F1 Score: {study.best_value}")
