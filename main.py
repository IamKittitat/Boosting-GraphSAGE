import os
import yaml
import numpy as np
import pandas as pd
import numpy as np
from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction
from src.train_graphsage import train_graphsage, train_boosting_graphsage
from src.load_config import load_config

def main():
    config = load_config()
    
    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['features_file']), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, config['file_path']['labels_file']), header=None)
    labels = labels.to_numpy().flatten()

    # Calculate Distance matrix
    distance_matrix = cal_distance_matrix(features, config['data']['dissimilarity_measure'])
    np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['distance_matrix_output']), distance_matrix, delimiter=",")

    # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
    distance_threshold = cal_distance_threshold(distance_matrix)
    tau_sick, tau_healthy = cal_neighbor_threshold(labels, config['model']['perc_val'], config['data']['is_balanced'])

    # Get Graph (Adjacency Matrix) from the distance matrix and threshold
    adj_matrix = md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels)
    np.savetxt(os.path.join(CURRENT_DIR, config['file_path']['adj_matrix_output']), adj_matrix, delimiter=",", fmt="%d")

    # Preparing train/val/test | or stratify k-fold cv (k = 10)
    # train_graphsage(features, adj_matrix, labels, config)
    train_boosting_graphsage(features, adj_matrix, labels, config)

if __name__ == "__main__":
    main()