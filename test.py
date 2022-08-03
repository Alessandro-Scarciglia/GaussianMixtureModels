import matplotlib.pyplot as plt

from GMM import KMeans
from utils import generate_gaussian_clusters
import numpy as np


if __name__ == "__main__":

    # Generate three clusters with Gaussian PDF
    model = KMeans()
    means = np.array([[1.]*3, [5.]*3])
    vars = np.array([.2*np.eye(3), 3*np.eye(3)])
    points = generate_gaussian_clusters(means, np.sqrt(vars), 100)

    plt.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()

