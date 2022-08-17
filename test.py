import matplotlib.pyplot as plt

from GMM import KMModel
from utils import generate_gaussian_clusters
import numpy as np

CLUSTERS = 3


if __name__ == "__main__":

    # Generate three clusters with Gaussian PDF
    #np.random.seed(12321)
    means = np.random.random((CLUSTERS, 2))*15
    variances = [2*np.eye(2) for _ in range(CLUSTERS)]
    points = generate_gaussian_clusters(means, np.sqrt(variances), 150)

    # Test the representative choice
    mixture = points.reshape((points.shape[0]*points.shape[1], -1))
    model = KMModel(points=mixture, k=CLUSTERS)

    zs, mix = model.run(500)

    for key in mix.keys():
        mix[key] = np.array(mix[key])

    # Show estimates
    plt.figure(0)
    plt.title("Estimate")
    for cluster in range(len(mix)):
        plt.scatter(mix[str(cluster)][:, 0], mix[str(cluster)][:, 1])
        plt.scatter(zs[int(cluster), 0], zs[int(cluster), 1], c='black', marker="^")
        plt.annotate(cluster, (zs[int(cluster), 0], zs[int(cluster), 1]))

    # Show Truth
    plt.figure(1)
    for k in range(len(points)):
        plt.scatter(points[k, :, 0], points[k, :, 1])

    plt.show()
