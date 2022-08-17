import numpy as np


def generate_gaussian_clusters(means: np.ndarray,
                               sigmas: np.ndarray,
                               point_per_cluster: int):
    """
    :param means: a numpy arrays with the means of the clusters
    :param sigmas: a numpy arrays with the std of the clusters
    :param point_per_cluster: number of classes
    :return: a single mixture k * N * d (clusters * No. samples * Dimension)
    """
    cluster = list()
    for mu, sigma in zip(means, sigmas):
        cluster.append(np.random.multivariate_normal(mean=mu,
                                                     cov=sigma,
                                                     size=point_per_cluster))

    return np.array(cluster)

