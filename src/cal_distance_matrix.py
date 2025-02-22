from scipy.spatial.distance import cdist

def cal_distance_matrix(features, measurement):
    """
    Compute the distance matrix for the given features using the specified measurement.

    Args:
    - features: A 2D numpy array (N, f) where N is the number of samples and f is the number of features.
    - measurement: A string specifying the type of distance ('EUCLIDEAN', 'COSINE', 'MANHATTAN').

    Returns:
    - A 2D numpy array of shape (N, N) representing the pairwise distances between samples.
    """
    # Check for valid measurement types
    if measurement == "EUCLIDEAN":
        return cdist(features, features, metric='euclidean')
    elif measurement == "COSINE":
        return cdist(features, features, metric='cosine')
    elif measurement == "MANHATTAN":
        return cdist(features, features, metric='cityblock')
    else:
        raise ValueError("Measurement not supported")

