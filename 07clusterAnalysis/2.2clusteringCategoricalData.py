import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans

data = pd.read_csv("2.0countryClusters.csv")

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})

#! using just language for clustering

x = data_mapped.iloc[:,3:4] # only language column stays

#* clustering
# kmeans = KMeans(3)
# kmeans.fit(x)

#* clustering results
# identifed_clusters = kmeans.fit_predict(x)
# data_with_clusters = data.copy()
# data_with_clusters['Cluster'] = identifed_clusters
# print(identifed_clusters) 
# result: usa+uk+canada+australia vs germany vs france

#* ploting clusters (grouped by only language)

# plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.show()


#! using all data for clustering (geography + language)

x_all = data_mapped.iloc[:,1:4]

kmeans = KMeans(3)
kmeans.fit(x_all)

#* clustering results
identifed_clusters_all = kmeans.fit_predict(x_all)
data_with_clusters_all = data.copy()
data_with_clusters_all['Cluster'] = identifed_clusters_all
# america vs europe vs australia

#* ploting clusters

plt.scatter(data_with_clusters_all['Longitude'], data_with_clusters_all['Latitude'], c=data_with_clusters_all['Cluster'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()