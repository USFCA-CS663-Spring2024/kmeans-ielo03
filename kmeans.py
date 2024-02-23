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
    all_distances = np.zeros((centers.shape[0], X.shape[0]), dtype=np.float64)
    for i, center in enumerate(centers):
        distances = np.sqrt(np.sum((X - center) ** 2, axis=1))
        all_distances[i] = distances
        mask = distances < min_distances
        closest_centers[mask] = i
        min_distances[mask] = distances[mask]
    return closest_centers, all_distances

def balance_closest(X, closest_centers, all_distances):
    cluster_counts = np.bincount(closest_centers, minlength=all_distances.shape[0])
    max_count = np.max(cluster_counts)
    min_count = np.min(cluster_counts)
    mean = np.mean(cluster_counts)
    while ((max_count - min_count) > 2):
        i = np.argmin(cluster_counts)
        distances = all_distances[i].copy()
        while (cluster_counts[i] < mean - 1):
            shortest_distance_i = np.argmin(distances)
            if (closest_centers[shortest_distance_i] != i and cluster_counts[closest_centers[shortest_distance_i]] >= mean):
                closest_centers[shortest_distance_i] = i
                cluster_counts[closest_centers[shortest_distance_i]] -= 1
                cluster_counts[i] += 1
            distances[shortest_distance_i] = np.inf
        max_count = np.max(cluster_counts)
        min_count = np.min(cluster_counts)
    return closest_centers

def find_new_centers(X, closest_centers, k):
    centers = np.zeros((k, X.shape[1]), dtype=np.float64)
    for i in range(k):
        closest_to_k = X[closest_centers == i]
        if len(closest_to_k) > 0:
            centers[i] = np.mean(closest_to_k, axis=0)
    return centers

class kmeans(cluster):
    def __init__(self, k=5, max_iterations=100, balanced=False):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced;

    def fit(self, X):
        X = np.array(X)
        centers = initialize_cluster_centers(X, self.k)
        closest_centers = find_closest_centers(X, centers)
        for i in range(self.max_iterations):
            centers = find_new_centers(X, closest_centers, self.k)
            closest_centers, all_distances = find_closest_centers(X, centers)
            if (self.balanced):
                closest_centers = balance_closest(X, closest_centers, all_distances)
        cluster_counts = np.bincount(closest_centers, minlength=all_distances.shape[0])
        return closest_centers, centers, cluster_counts
            