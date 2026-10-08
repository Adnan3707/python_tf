import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN


df = pd.DataFrame({
    'Invoice ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Amount': [85, 92, 91, 87, 90, 94, 95, 60, 89, 120, 129, 450]
})

# Cluster the invoice amounts; DBSCAN labels outliers as -1.
df['Cluster'] = DBSCAN(eps=20, min_samples=2).fit_predict(df[['Amount']])

fig, ax = plt.subplots(figsize=(9, 5))
for cluster in sorted(df['Cluster'].unique()):
    points = df[df['Cluster'] == cluster]
    label = 'Noise / anomaly' if cluster == -1 else f'Cluster {cluster}'
    ax.scatter(points['Invoice ID'], points['Amount'], label=label, s=80)

ax.set_title('DBSCAN Clustering of Invoice Amounts')
ax.set_xlabel('Invoice ID')
ax.set_ylabel('Amount')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()