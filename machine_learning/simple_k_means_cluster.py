import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset with 3 clusters
np.random.seed(42)
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in range(3):
    plt.scatter(
        X[labels == cluster, 0], X[labels == cluster, 1], label=f"Cluster {cluster + 1}"
    )
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering Example")
plt.legend()
plt.grid(True)
plt.show()
