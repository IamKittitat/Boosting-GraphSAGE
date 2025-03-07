import os
import pandas as pd
import optuna
from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.train_graphsage import train_graphsage
from src.load_config import load_config

def objective(trial):
    config = load_config()
    
    dissimilarity_measure = trial.suggest_categorical('dissimilarity_measure', ['EUCLIDEAN', 'COSINE', 'MANHATTAN'])
    perc_val = trial.suggest_float('perc_val', 5, 100, step=5)
    
    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()

    # Calculate Distance matrix
    distance_matrix = cal_distance_matrix(features, dissimilarity_measure)
    
    # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
    distance_threshold = cal_distance_threshold(distance_matrix)
    tau_sick, tau_healthy = cal_neighbor_threshold(labels, perc_val, config['data']['is_balanced'])

    # Get Graph (Adjacency Matrix) from the distance matrix and threshold
    adj_matrix = md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels)

    val_f1, test_f1 = train_graphsage(features, adj_matrix, labels, config)
    
    return val_f1

study = optuna.create_study(direction='maximize')  # We want to maximize the accuracy
study.optimize(objective, n_trials=50)  # Run for 50 trials

print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation F1 score: {study.best_value}")
