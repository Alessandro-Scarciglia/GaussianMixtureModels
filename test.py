import matplotlib.pyplot as plt
from GMM import KMModel
from utils import generate_gaussian_clusters
import numpy as np

# Parameters
CLUSTERS = 5

# Main
if __name__ == "__main__":

    # Generate clusters with Gaussian PDF: this step is not part of the algorithm but it is necessary to generate
    # a synthetic set of clusters to test GMMs.
    means = np.random.random((CLUSTERS, 2)) * 30
    variances = [40 * np.eye(2) for _ in range(CLUSTERS)]
    points = generate_gaussian_clusters(means, np.sqrt(variances), 100)
    mixture = points.reshape((points.shape[0] * points.shape[1], -1))

    # Clustering with a GMM model
    model = KMModel(points=mixture, k=CLUSTERS)  # EMModel coming soon!
    zs, mix = model.run(n_iterations=100)

    # Transform the outcome in numpy array format
    for key in mix.keys():
        mix[key] = np.array(mix[key])

    # Plot estimates
    plt.figure(0)
    plt.title("Estimate")
    for cluster in range(len(mix)):
        plt.scatter(mix[str(cluster)][:, 0], mix[str(cluster)][:, 1])
        plt.scatter(zs[int(cluster), 0], zs[int(cluster), 1], c='black', marker="^")
        plt.annotate(cluster, (zs[int(cluster), 0], zs[int(cluster), 1]))

    # Plot ground-truths
    plt.figure(1)
    for k in range(len(points)):
        plt.scatter(points[k, :, 0], points[k, :, 1], c='blue')

    plt.show()
