import sys
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from pysml import kmeans


if __name__ == '__main__':
    run = {
        'kmeans': kmeans
    }['kmeans' if len(sys.argv) < 2 else sys.argv[1]]

    X, yt = make_blobs(centers=4, n_samples=1000, cluster_std=0.8)
    est = run(X, n_clusters=4)
    print('score:', est.score(X))

    plt.scatter(X[:, 0], X[:, 1], c=est.labels, s=50, cmap='viridis')
    plt.scatter(est.centers[:, 0], est.centers[:, 1], c='black', s=250, alpha=0.6)
    plt.show()
