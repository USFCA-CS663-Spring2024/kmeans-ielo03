from cluster import cluster
import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def initialize_cluster_centers(X, k):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    centers = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
    return centers

def find_closest_centers(X, centers):
    closest_centers = np.zeros(X.shape[0], dtype=int)
    min_distances = np.full(X.shape[0], np.inf)

    for i, center in enumerate(centers):
        distances = np.sqrt(np.sum((X - center) ** 2, axis=1))
        mask = distances < min_distances
        closest_centers[mask] = i
        min_distances[mask] = distances[mask]

    return closest_centers

def find_new_centers(X, closest_centers, k):
    centers = np.zeros((k, X.shape[1]), dtype=np.float64)
    for i in range(k):
        closest_to_k = X[closest_centers == i]
        if len(closest_to_k) > 0:
            centers[i] = np.mean(closest_to_k, axis=0)
    return centers

class kmeans(cluster):
    def __init__(self, k=5, max_iterations=100):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        X = np.array(X)
        centers = initialize_cluster_centers(X, self.k)
        closest_centers = find_closest_centers(X, centers)
        for i in range(self.max_iterations):
            centers = find_new_centers(X, closest_centers, self.k)
            closest_centers = find_closest_centers(X, centers)
        return closest_centers, centers
            