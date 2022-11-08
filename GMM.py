import numpy as np

"""K-Means model"""


class KMModel:
    def __init__(self, points: np.ndarray, k: int):
        self.points = points
        self.k = k
        self.zs = self.__initialize_representatives(points, k)
        self.clusters = self.__initialize_clusters(k)

    @staticmethod
    def __initialize_representatives(points, k):
        index_array = [i for i in range(points.shape[0])]
        selected_indexes = np.random.choice(a=index_array, size=k, replace=False)

        return points[selected_indexes]

    @staticmethod
    def __initialize_clusters(k):
        clusters = dict()
        for i in range(k):
            clusters[str(i)] = []

        return clusters

    def one_step_run(self):
        # STEP 0: Zero the clusters
        self.clusters = self.__initialize_clusters(k=self.k)

        # Step 1: Assign each point to the closest representative
        for point in self.points:
            squared_euclidean_distance = list()
            for z in self.zs:
                squared_euclidean_distance.append(np.linalg.norm(point - z) ** 2)
            self.clusters[str(np.argmin(squared_euclidean_distance))].append(point)

        # Step 2: Update representatives with the closest to the mean value
        for bucket in self.clusters.keys():
            # Look for the closest z to zs
            distances = np.linalg.norm(self.clusters[bucket] - np.mean(self.clusters[bucket], axis=0), axis=1)
            closest_index = np.argmin(distances)
            self.zs[int(bucket)] = self.clusters[bucket][closest_index]

        return self.zs, self.clusters

    def run(self, n_iterations: int):
        # Iterate for N steps
        for _ in range(n_iterations):
            self.one_step_run()

        return self.zs, self.clusters


"""Expectation-Maximization Model"""


class EMModel:

    @staticmethod
    def __initialize_pjs(k):
        """Mixture probabilities are drawn from a uniform distribution"""
        return np.random.uniform(low=0, high=1, size=k)

    @staticmethod
    def __initialize_vars(k, points):
        """Variances are initialized as the global set variance"""
        sigma = np.std(points)
        return np.array([sigma ** 2] * k)

    @staticmethod
    def __initialize_means(k, points):
        """Means are initialized by K-means"""
        # TODO: Implement initialization
        pass

    def __expectation_step(self):
        # TODO: Implement the E-step
        pass

    def __maximization_step(self):
        # TODO: Implement the M-step
        pass

    def run(self, training_set: np.ndarray, num_clusters: int):
        """Initialization of parameters"""
        # Initialization of mixtures probabilities
        pj = self.__initialize_pjs(num_clusters)
        variances = self.__initialize_vars(num_clusters, training_set)
        means = self.__initialize_means(num_clusters, training_set)
