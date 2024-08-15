import numpy as np
from sklearn.neighbors import NearestNeighbors


def logarithm(x, t):
    """
    Compute the logarithm of the sum of x and t.

    This function calculates the natural logarithm (base e) of the quantity (x + t),
    where x and t can be floats, lists, numpy arrays, or pandas Series. It is useful
    for transforming data in various numerical and statistical applications.

    Args:
        x (Union[float, list, np.ndarray, pd.Series]): The first input value(s).
        t (float): The second input value to be added to x. Should be a non-negative number.

    Returns:
        Union[float, np.ndarray, pd.Series]: The natural logarithm of the sum of x and t.
    """
    return np.log(x + t)


def soft_min(scores, gamma):
    """
    Computes the soft minimum of a list of scores.

    The soft minimum is a differentiable approximation of the minimum function,
    controlled by the parameter gamma. Higher values of gamma make the approximation
    closer to the actual minimum.

    Parameters:
    scores (array-like): A list or array of numerical scores.
    gamma (float): A positive parameter that controls the sharpness of the approximation.
                   Higher values result in a closer approximation to the minimum.

    Returns:
    float: The soft minimum of the scores.
    """
    softmin = (-1 / gamma) * (np.log((1 / (len(scores) - 1)) * np.sum(np.exp(-gamma * scores))))
    return softmin


def compute_anomaly_scores(train_data, test_data, gamma):
    """
    Computes anomaly scores for the test data based on the training data using
    a soft minimum approach.

    This function fits a nearest neighbors model on the training data and then
    calculates distances to nearest neighbors for the test data. It uses a soft
    minimum function to compute anomaly scores, which provide an indication of
    how anomalous each test data point is.

    Parameters:
    train_data (array-like): The training data used to fit the nearest neighbors model.
    test_data (array-like): The test data for which anomaly scores are to be computed.
    gamma (float): A positive parameter for the soft minimum function, controlling the
                   sharpness of the approximation. Higher values make the approximation
                   closer to the actual minimum.

    Returns:
    np.ndarray: An array of anomaly scores for the test data.
    """
    # Fit nearest neighbors model on the training data
    nbrs = NearestNeighbors(n_neighbors=len(train_data), algorithm="ball_tree").fit(train_data)

    # Calculate distances to nearest neighbors on the test data
    distances, _ = nbrs.kneighbors(test_data)

    # Calculate softmin-based anomaly scores
    anomaly_scores = []
    for i in range(len(distances)):
        scores = distances[i][:-1]
        anomaly_score = soft_min(scores, gamma)
        anomaly_scores.append(anomaly_score)

    return np.array(anomaly_scores)


def compute_anomaly_contributions(data, gamma, anomaly_scores):
    """
    Compute the anomaly contributions for each data point.

    This function calculates the anomaly contributions for each data point in the dataset
    based on the method described in:
    "Explaining the Predictions of Unsupervised Learning Models" by G. Montavon, J. R. Kauffmann,
    W. Samek, and K.-R. Müller, volume 13200 of Lecture Notes in Computer Science, pages 117–138, Springer, 2020.

    Args:
        data (np.ndarray): The dataset, a 2D array where each row represents a data point.
        gamma (float): The gamma parameter used in the exponential calculation.
        anomaly_scores (np.ndarray): The anomaly scores for each data point.

    Returns:
        np.ndarray: A 2D array where each element (i, j) represents the anomaly contribution of data point j to data point i.
    """
    # Initialize array to store anomaly contributions for each data point
    anomaly_contributions = np.zeros((len(data), len(data)))

    # Compute anomaly contributions for each data point
    for j, instance_j in enumerate(data):
        # Calculate zjk values for all data points other than j
        zjk_values = np.exp(-gamma * np.sum((data - instance_j) ** 2, axis=1))

        # Calculate denominator for normalization
        denominator = np.sum(zjk_values)

        # Calculate anomaly contributions for each data point other than j
        for k, zjk in enumerate(zjk_values):
            anomaly_contributions[j, k] = zjk / denominator * anomaly_scores[k]

    return anomaly_contributions


def propagate_feature_contributions(anomaly_contributions, data):
    """
    Propagate feature contributions for each data point based on anomaly contributions.

    This function calculates the feature contributions for each data point in the dataset
    based on the method described in:
    "Explaining the Predictions of Unsupervised Learning Models" by G. Montavon, J. R. Kauffmann,
    W. Samek, and K.-R. Müller, volume 13200 of Lecture Notes in Computer Science, pages 117–138, Springer, 2020.

    Args:
        anomaly_contributions (np.ndarray): The anomaly contributions for each data point.
        data (np.ndarray): The dataset, a 2D array where each row represents a data point.

    Returns:
        np.ndarray: A 2D array where each element (i, j) represents the contribution of feature j to the anomaly score of data point i.
    """
    # Initialize array to store feature contributions for each data point
    feature_contributions = np.zeros((len(data), data.shape[1]))

    # Compute feature contributions for each data point
    for j, instance_j in enumerate(data):
        # Initialize array to store squared Euclidean distances for each feature
        feature_distances = np.zeros((len(data), data.shape[1]))

        # Calculate squared Euclidean distances between instance j and all other instances for each feature
        for i in range(data.shape[1]):
            feature_distances[:, i] = (data[:, i] - instance_j[i]) ** 2

        # Calculate denominator for normalization
        denominator = np.sum(feature_distances * anomaly_contributions[j][:, np.newaxis])

        # Calculate feature contributions for instance j
        for i in range(data.shape[1]):
            numerator = np.sum(feature_distances[:, i] * anomaly_contributions[j])
            feature_contributions[j, i] = numerator / denominator

    return feature_contributions
