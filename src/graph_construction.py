import numpy as np

def edge_construction(adjacency_matrix, distance_matrix, distance_threshold, tau, v_i, N):
    new_adjacency_matrix = adjacency_matrix.copy()
    sorted_indices = np.argsort(distance_matrix[v_i])
    
    current_neighbors = 0
    for idx in sorted_indices:
        if distance_matrix[v_i, idx] <= distance_threshold and current_neighbors < tau:
            new_adjacency_matrix[v_i, idx] = 1
            current_neighbors += 1
    
    return new_adjacency_matrix, current_neighbors

def refine_edges(adjacency_matrix, distance_matrix, distance_threshold, tau, sparse_nodes, N):
    refined_adjacency_matrix = adjacency_matrix.copy()
    delta = (np.max(distance_matrix.flatten()) - distance_threshold) / 5
    
    for v_i in sparse_nodes:
        new_distance_threshold = distance_threshold + delta
        current_neighbors = 0
        while(current_neighbors < tau):    
            refined_adjacency_matrix, current_neighbors = edge_construction(refined_adjacency_matrix, distance_matrix, new_distance_threshold, tau, v_i, N)
            new_distance_threshold += delta

    return refined_adjacency_matrix


def md_graph_construction(distance_matrix, distance_threshold, tau_sick, tau_healthy, labels):
    """
    Construct the MD-Graph based on the provided distance matrix and thresholds.

    Parameters:
    - distance_matrix: 2D numpy array of shape (N, N) representing the pairwise distances between samples.
    - distance_threshold: The maximum threshold distance for edge construction.
    - tau_sick: Maximum number of neighbors for sick nodes.
    - tau_healthy: Maximum number of neighbors for healthy nodes.

    Returns:
    - adjacency_matrix: Numpy array of shape (N, N) representing the adjacency matrix of the MD-Graph.
    """
    N = distance_matrix.shape[0]
    adjacency_matrix = np.zeros((N, N), dtype=int)
    sparse_nodes = []

    # Edge Construction
    for v_i in range(N):
        tau = tau_sick if labels[v_i] == 1 else tau_healthy
        adjacency_matrix, current_neighbors = edge_construction(adjacency_matrix, distance_matrix, distance_threshold, tau, v_i, N)
        if(current_neighbors == 0):
            sparse_nodes.append(v_i)
    
    # Edge Refinement
    adjacency_matrix = refine_edges(adjacency_matrix, distance_matrix, distance_threshold, tau, sparse_nodes, N)
    
    return adjacency_matrix