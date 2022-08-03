import numpy as np

"""K-Means model"""


class KMeans:

    @staticmethod
    def __initialize_representatives(points, k):
        return np.random.choice(points, k, replace=False)

    @staticmethod
    def bho():
        pass

    def train(self, training_set: np.ndarray, num_clusters: int):
        zs = self.__initialize_representatives(training_set, num_clusters)
        return zs


"""Expectation-Maximization Model"""


class EMModel:

    @staticmethod
    def __initialize_pjs(k):
        """Mixture probabilities are drawn from a uniforn distribution"""
        return np.random.uniform(low=0, high=1, size=k)

    @staticmethod
    def __initialize_vars(k, points):
        """Variances are initialized as the global set variance"""
        sigma = np.std(points)
        return np.array([sigma ** 2] * k)

    @staticmethod
    def __initialize_means(k, points):
        """Means are initialized by K-means"""
        return 0

    def __expectation_step(self):
        pass

    def __maximization_step(self):
        pass

    def train(self, training_set: np.ndarray, num_clusters: int):
        """Initialization of parameters"""
        # Initialization of mixtures probabilities
        pj = self.__initialize_pjs(num_clusters)
        vars = self.__initialize_vars(num_clusters, training_set)
        means = self.__initialize_means(num_clusters, training_set)
