# choose the number of clusters (k)
# specify the cluster seeds (starting centroids) (randomly or using prior knowledge abt the data)
# assign each point to a centroid based on how close the point and centroid are.
# adjust the centroids. then recalculate and asign points to these centroids by distance.
# if any point's centroid is changed, adjust the centroids again. do this till all points are closer to their centroids than any other centroid.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans

data = pd.read_csv("2.0countryClusters.csv")

#* ploting the data of countries based on their geographical location
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#* select the features(latitude and longitude-enlem boylam-)
x = data.iloc[:,1:3] # slices the data frame, given rows and columns to be kept. in this case all rows will be kept, and columns 1 and 2 will be shown. (latitude and longitude)
print(x)

#* clustering
kmeans = KMeans(3)
kmeans.fit(x)

#* clustering results
identifed_clusters = kmeans.fit_predict(x)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identifed_clusters
print(identifed_clusters) 
#2 cluster result: austrulia: cluster 0, others: cluster 1
#3 cluster result: austrilia vs usa+canada vs france+uk+germany

#* ploting clusters

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()