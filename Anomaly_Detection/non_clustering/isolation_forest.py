# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv('C:/Interview Process/python_tf/Anomaly_Detection/non_clustering/Fraud+Detection+Insurance+Non+Cluster+v1.csv')

customer_ids = df['customer_id']

df = df.drop(columns=['customer_id'])

numeric_colums = df.select_dtypes(include=['float64','int64']).columns
non_numeric_colums = df.select_dtypes(include=['object','bool','category']).columns

# print(numeric_colums)
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
        ('num',numeric_transformer,numeric_colums),
        ('cat',categorical_transformer,non_numeric_colums)
    ]
)
# %%
df_preprocessed = preprocessor.fit_transform(df)

# %%
df_preprocessed
# %%
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.01,random_state=42)
iso_forest.fit(df_preprocessed)
anomaly_labels = iso_forest.predict(df_preprocessed)

anomalies = anomaly_labels == -1

df['anomaly'] = anomalies
# %%
# output_path = 'C:/Interview Process/python_tf/Anomaly_Detection/non_clustering/insurance_anomalies.csv'
# df.to_csv(output_path, index=False)
# print(f"Saved anomaly results to: {output_path}")
# %%
