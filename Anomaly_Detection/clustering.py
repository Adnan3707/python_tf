import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN,KMeans


df = pd.read_csv('C:/Interview Process/python_tf/Anomaly_Detection/Cluster+Anomaly+Employee.csv')

numeric_features = df.select_dtypes(include=['int64','float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot',OneHotEncoder())
])

# print(numeric_features)
# Combine Processing Steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

df_preprocessed = preprocessor.fit_transform(df)

# print(df_preprocessed)
dbscan = DBSCAN(eps=4,min_samples=10)
db_label = dbscan.fit_predict(df_preprocessed)

# K-means

kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(df_preprocessed)
kmeans_labels=kmeans.predict(df_preprocessed)
kmeans_distances = kmeans.transform(df_preprocessed).min(axis=1)
kmeans_threashold = np.percentile(kmeans_distances,95)
kmeans_anomalies = kmeans_distances > kmeans_threashold 
kmeans_labels_with_anomalies = np.where(kmeans_anomalies,-1,kmeans_labels)



print("Number of anomalies detected by each method:")
print(f"DBSCAN: {np.sum(db_label == -1)}")
print(f"KMeans: {np.sum(kmeans_labels_with_anomalies == -1)}")


from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


# Project the encoded features to two dimensions for visualization.
plot_data = TruncatedSVD(n_components=2, random_state=42).fit_transform(df_preprocessed)

# DBSCAN plot
scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=db_label, cmap='tab10', s=30, alpha=0.7)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('DBSCAN Clusters (-1 = noise)')
plt.colorbar(scatter, label='Cluster label')
plt.tight_layout()
plt.show()

# K-means plot with anomalies highlighted
anomaly_mask = kmeans_labels_with_anomalies == -1
plt.figure()
plt.scatter(plot_data[~anomaly_mask, 0], plot_data[~anomaly_mask, 1], c=kmeans_labels[~anomaly_mask], cmap='tab10', s=30, alpha=0.7)
plt.scatter(plot_data[anomaly_mask, 0], plot_data[anomaly_mask, 1], c='red', s=50, marker='x', label='Anomaly')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('KMeans Clusters with Anomalies')
plt.colorbar(label='Cluster label')
plt.legend()
plt.tight_layout()
plt.show()


