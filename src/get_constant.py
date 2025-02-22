import numpy as np

def cal_distance_threshold(distance_matrix):
    flat_matrix = distance_matrix.flatten()
    return np.median(flat_matrix)

def cal_neighbor_threshold(labels, perc_val, is_balanced = True):
    total_samples = len(labels)
    sick_samples = np.sum(labels)
    healthy_samples = total_samples - sick_samples

    if(is_balanced):
        tau = (perc_val * total_samples) / 100
        return tau, 0
    else:
        tau_sick = np.floor((perc_val * sick_samples) / 100)
        tau_healthy = np.floor((perc_val * healthy_samples) / 100)
        return tau_sick, tau_healthy

    