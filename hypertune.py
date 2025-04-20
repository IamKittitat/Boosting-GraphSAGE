import optuna
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.train_graphsage import train_boosting_graphsage, train_gradient_boosting_graphsage
from src.load_config import load_config
from torch_geometric.utils import dense_to_sparse

def objective(trial):
    config = load_config()

    config['data']['dissimilarity_measure'] = trial.suggest_categorical('dissimilarity_measure', ['EUCLIDEAN', 'COSINE', 'MANHATTAN']) # 3
    config['model']['perc_val'] =  trial.suggest_int('perc_val', 5, 50, step=5) #10
    config['model']['embed_dim'] = trial.suggest_categorical('embed_dim', [2**x for x in range(5, 9)]) #4
    config['model']['lr'] =trial.suggest_categorical('lr', [0.001, 0.005, 0.01, 0.05, 0.1]) #5
    config['model']['base_estimators'] = trial.suggest_int('base_estimators', 5, 50, step=5) #10
    config['model']['epoch'] = trial.suggest_int('epoch', 50, 100, step=10) #16
    config['model']['num_layers'] = trial.suggest_int('num_layers', 2, 6) #5
    # lr_boost = trial.suggest_categorical('lr_boost', [0.001, 0.005, 0.01, 0.05, 0.1])

    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()

    # 10-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(features)), labels)):
        print(f"Fold {fold+1}")
        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)

        # Calculate Distance Matrix
        distance_matrix_train = cal_distance_matrix(features[train_idx], config['data']['dissimilarity_measure'])
        np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['distance_matrix_output']), 
                   distance_matrix_train, delimiter=",")

        # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
        distance_threshold = cal_distance_threshold(distance_matrix_train)
        tau_sick, tau_healthy = cal_neighbor_threshold(labels[train_idx], config['model']['perc_val'], config['data']['is_balanced'])
        
        # Get Graph (Adjacency Matrix) from the distance matrix and threshold
        adj_matrix_train = md_graph_construction(distance_matrix_train, distance_threshold, tau_sick, tau_healthy, labels[train_idx])
        np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['adj_matrix_output']), 
                   adj_matrix_train, delimiter=",", fmt="%d")

        features_torch = torch.FloatTensor(features)
        adj_matrix_train_torch = torch.FloatTensor(adj_matrix_train)
        labels_torch = torch.LongTensor(labels)   
        edge_index_train, _ = dense_to_sparse(adj_matrix_train_torch)

        # Calculate Distance Matrix for each validation node
        neighbor_val = []
        for val_node in val_idx:
            distance_matrix_val = cal_distance_matrix(
                np.concatenate([features[train_idx], features[val_node].reshape(1,-1)], axis=0), 
                config['data']['dissimilarity_measure'])[-1, :-1] # Get only distance matrix of validation node (Exclude self)

            tau_node = tau_healthy if labels[val_node] == 0 else tau_sick
            neighbor_node = np.argsort(distance_matrix_val)[:int(tau_node)]
            neighbor_val.append(neighbor_node)

        features_train = features_torch[train_idx]
        features_val = features_torch[val_idx]
        labels_train = labels_torch[train_idx]
        labels_val = labels_torch[val_idx]

        # Train the model and calculate AUC score
        _, val_auc = train_boosting_graphsage(features_train, features_val, adj_matrix_train_torch, labels_train, labels_val, 
                                           edge_index_train, neighbor_val, embed_dim=config['model']['embed_dim'], 
                                           lr=config['model']['lr'], base_estimators=config['model']['base_estimators'],
                                           num_epochs=config['model']['epoch'], num_layers=config['model']['num_layers'])
        auc_scores.append(val_auc)

    avg_auc = sum(auc_scores) / len(auc_scores)
    return avg_auc

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best AUC: {study.best_value:.4f}")

if __name__ == "__main__":
    main()
