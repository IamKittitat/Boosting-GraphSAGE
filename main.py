import os
import numpy as np
import pandas as pd

from src.cal_distance_matrix import cal_distance_matrix
from src.get_constant import cal_distance_threshold, cal_neighbor_threshold
from src.graph_construction import md_graph_construction

# CONSTANT
# features name
# MEASUREMENT 
# PERC_VAL


def main():
    # Get OTU Features
    CURRENT_DIR = os.path.dirname(__file__)
    features = pd.read_csv(os.path.join(CURRENT_DIR, "data/2_OTU/GDMicro_T2D_features.csv"), header=None)
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(CURRENT_DIR, "data/2_OTU/GDMicro_T2D_labels.csv"), header=None)
    labels = labels.to_numpy().flatten()

    # Calculate Distance matrix
    distance_matrix = cal_distance_matrix(features, "EUCLIDEAN")
    np.savetxt(os.path.join(CURRENT_DIR, "data/3_distance_matrix/GDMicro_T2D.csv"), distance_matrix, delimiter=",")

    # Prepare Distance Threshold, Neighbor Threshold (tau_sick, tau_healthy)
    distance_threshold = cal_distance_threshold(distance_matrix)
    tau_sick, tau_healthy = cal_neighbor_threshold(labels, perc_val=20, is_balanced=False)

    # Create graph -> (distance matrix, threshold, tau_sick, tau_healthy) -> Adj matrix
        # For loop every vertex and call edge_construction (... + v) -> adj matrix
        # Edge refinement 
    adj_matrix = md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels)
    np.savetxt(os.path.join(CURRENT_DIR, "data/4_adj_matrix/GDMicro_T2D.csv"), adj_matrix, delimiter=",", fmt="%d")
    
    # Prepare feature vector, adj, label


    # Preparing train/val/test | or stratify k-fold cv (k = 10)
        # Train model -> (feat_vec, adh, label) -> model + result?
        # Evaluate model
        # Log results
        # Save model
    return


if __name__ == "__main__":
    main()