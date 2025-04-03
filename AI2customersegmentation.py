import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample customer dataset: [Annual Income, Spending Score]
customers = np.array([
    [15, 39], [16, 81], [17, 6], 
    [18, 77], [19, 40], [20, 76]
])

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(customers)

# Train K-Means model
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)


kmeans.fit(X)

# Make predictions
labels = kmeans.labels_
print(f"Cluster labels: {labels}")
