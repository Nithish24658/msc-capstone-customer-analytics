import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Data
df = pd.read_csv('data/customer_raw_data.csv')

# Engineering 25+ CRM/Clickstream Proxy Metric Matrix
features_to_cluster = ['recency', 'frequency', 'monetary_value', 'web_sessions', 'click_through_rate']
X = df[features_to_cluster]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
print("Clustering data into 5 distinct personas...")
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
df['persona_cluster'] = kmeans.fit_predict(X_scaled)

# Evaluation profiles
cluster_summary = df.groupby('persona_cluster')[features_to_cluster].mean()
print("\nGenerated Persona Cluster Metric Matrix:\n", cluster_summary)

# Save segmented records
df.to_csv('data/customer_segmented_data.csv', index=False)
