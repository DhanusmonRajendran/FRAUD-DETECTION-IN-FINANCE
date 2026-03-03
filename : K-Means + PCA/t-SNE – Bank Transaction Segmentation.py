import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

data = pd.read_csv('bank_transactions_data_2.csv')
df_int = data.select_dtypes(include=['int64','float64'])
df_std = pd.DataFrame(StandardScaler().fit_transform(df_int),
                      columns=df_int.columns)
kmeans4 = KMeans(n_clusters=4, n_init='auto', random_state=42)
labels4 = kmeans4.fit_predict(df_std)
print('Silhouette (k=4):', silhouette_score(df_std, labels4))
# Cluster 0: High login attempts  Cluster 1: Young low-balance
# Cluster 2: Senior high-balance  Cluster 3: High-amount txns
