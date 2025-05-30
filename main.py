import os
import torch
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.train_graphsage import train_graphsage, train_boosting_graphsage, train_gradient_boosting_graphsage
from src.load_config import load_config
from torch_geometric.utils import dense_to_sparse

def main():
    config = load_config()
    
    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()

    # 10 fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_auc_scores = []
    val_auc_scores = []

#     train_idx, val_idx = train_test_split(
#     np.arange(len(features)),
#     test_size=0.3,
#     stratify=labels,
#     random_state=42
# )
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(features)), labels)):
        print(f"Fold {fold+1}")
        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)

        # Calculate Distance matrix
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

        # Cal Distance Matrix for each val node
        neighbor_val = []
        for val_node in val_idx:
            distance_matrix_val = cal_distance_matrix(
                np.concatenate([features[train_idx], features[val_node].reshape(1,-1)], axis=0), 
                config['data']['dissimilarity_measure'])[-1, :-1] # Get only distance matrix of val node (Exclude self)

            tau_node = tau_healthy if labels[val_node] == 0 else tau_sick
            neighbor_node = np.argsort(distance_matrix_val)[:int(tau_node)]
            neighbor_val.append(neighbor_node)

        features_train = features_torch[train_idx]
        features_val = features_torch[val_idx]
        labels_train = labels_torch[train_idx]
        labels_val = labels_torch[val_idx]

        # val_auc = train_graphsage(features_train, features_val, adj_matrix_train_torch, labels_train, labels_val, edge_index_train, 
        #                 neighbor_val, embed_dim=config['model']['embed_dim'], lr=config['model']['lr'], 
        #                 num_epochs=config['model']['epoch'], num_layers=config['model']['num_layers'])
        
        train_auc, val_auc = train_boosting_graphsage(features_train, features_val, adj_matrix_train_torch, labels_train, labels_val, 
                                           edge_index_train, neighbor_val, embed_dim=config['model']['embed_dim'], 
                                           lr=config['model']['lr'], base_estimators=config['model']['base_estimators'],
                                           num_epochs=config['model']['epoch'], num_layers=config['model']['num_layers'])
        train_auc_scores.append(train_auc)
        val_auc_scores.append(val_auc)

    avg_train_auc = sum(train_auc_scores) / len(train_auc_scores)
    avg_val_auc = sum(val_auc_scores) / len(val_auc_scores)
    
    print(f"Average Train AUC: {avg_train_auc:.4f}")
    print(f"Average Validation AUC: {avg_val_auc:.4f}")

if __name__ == "__main__":
    main()