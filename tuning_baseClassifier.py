import os
import torch
import pandas as pd
import numpy as np
import optuna
from optuna import Trial
from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.train_graphsage import train_graphsage, train_boosting_graphsage
from src.load_config import load_config

def objective(trial: Trial):
    config = load_config()

    base_estimators = trial.suggest_int('base_estimators', 5, 40, step=5)

    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()

    # Calculate Distance matrix using the sampled dissimilarity_measure
    distance_matrix = cal_distance_matrix(features, config['data']['dissimilarity_measure'])
    
    # Save distance matrix
    np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['distance_matrix_output']), distance_matrix, delimiter=",")

    # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
    distance_threshold = cal_distance_threshold(distance_matrix)
    tau_sick, tau_healthy = cal_neighbor_threshold(labels, config['data']['perc_val'], config['data']['is_balanced'])

    # Get Graph (Adjacency Matrix) from the distance matrix and threshold
    adj_matrix = md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels)
    
    # Save the adjacency matrix
    np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['adj_matrix_output']), adj_matrix, delimiter=",", fmt="%d")

    # Prepare data for model
    features = torch.FloatTensor(features)
    adj_matrix = torch.FloatTensor(adj_matrix)
    labels = torch.LongTensor(labels)

    avg_auc = train_boosting_graphsage(features, adj_matrix, labels, embed_dim=config['model']['embed_dim'], 
                                       lr=config['model']['lr'], num_epochs=config['model']['epoch'], 
                                       base_estimators=base_estimators, 
                                       num_layers=config['model']['num_layers'])

    return avg_auc

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best Trial:")
    print(f"  AUC: {study.best_value}")
    print(f"  Params: {study.best_params}")

if __name__ == "__main__":
    main()