import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('/content/Cluster Anomaly Employee.csv')

# Separate numeric and non-numeric columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
df_preprocessed = preprocessor.fit_transform(df)

# Convert to DataFrame for clustering
df_preprocessed = pd.DataFrame(df_preprocessed)

# DBSCAN
dbscan = DBSCAN(eps=4, min_samples=10)
dbscan_labels = dbscan.fit_predict(df_preprocessed)

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_preprocessed)
kmeans_labels = kmeans.predict(df_preprocessed)
kmeans_distances = kmeans.transform(df_preprocessed).min(axis=1)
kmeans_threshold = np.percentile(kmeans_distances, 95)  # 5% as anomalies
kmeans_anomalies = kmeans_distances > kmeans_threshold
kmeans_labels_with_anomalies = np.where(kmeans_anomalies, -1, kmeans_labels)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(df_preprocessed)
gmm_scores = gmm.score_samples(df_preprocessed)
gmm_threshold = np.percentile(gmm_scores, 5)  # Lower 5% as anomalies
gmm_anomalies = gmm_scores < gmm_threshold
gmm_labels_with_anomalies = np.where(gmm_anomalies, -1, gmm_labels)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(df_preprocessed)
hierarchical_distances = np.linalg.norm(df_preprocessed - kmeans.cluster_centers_[hierarchical_labels], axis=1)
hierarchical_threshold = np.percentile(hierarchical_distances, 95)  # 5% as anomalies
hierarchical_anomalies = hierarchical_distances > hierarchical_threshold
hierarchical_labels_with_anomalies = np.where(hierarchical_anomalies, -1, hierarchical_labels)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(df_preprocessed)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_preprocessed)

def plot_clusters(df_pca, labels, title):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            # Anomalies
            plt.scatter(df_pca[labels == label, 0], df_pca[labels == label, 1], s=50, label='Anomaly', color='red')
        else:
            plt.scatter(df_pca[labels == label, 0], df_pca[labels == label, 1], s=50, label=f'Cluster {label}')
    plt.title(title)
    plt.legend()
    plt.show()

# Plot results
plot_clusters(df_pca, dbscan_labels, 'DBSCAN Clusters and Anomalies')
plot_clusters(df_pca, kmeans_labels_with_anomalies, 'K-Means Clusters and Anomalies')
plot_clusters(df_pca, gmm_labels_with_anomalies, 'GMM Clusters and Anomalies')
plot_clusters(df_pca, hierarchical_labels_with_anomalies, 'Hierarchical Clustering and Anomalies')
plot_clusters(df_pca, lof_labels, 'LOF Clusters and Anomalies')

# Summarize results
print("Number of anomalies detected by each method:")
print(f"DBSCAN: {np.sum(dbscan_labels == -1)}")
print(f"K-Means: {np.sum(kmeans_anomalies)}")
print(f"GMM: {np.sum(gmm_anomalies)}")
print(f"Hierarchical Clustering: {np.sum(hierarchical_anomalies)}")
print(f"LOF: {np.sum(lof_labels == -1)}")