import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans

data = pd.read_csv("2.0countryClusters.csv")

#! elbow method for choosing the number of clusters

"""
our two goals: minimizing the distance between points in a cluster === maximizing the distance between clusters

distance between points in a cluster: "within-cluster sum of squares: WCSS"
so minimizing WCSS = the perfect clustering solution

*optimal number of clusters is 3, it can be shown in a graph looking like an elbow, which has given the name of the method.

"""

data_mapped = data.copy()
data_mapped['Language']=data_mapped['Language'].map({'English':0,'French':1,'German':2})
data_mapped

x = data_mapped.iloc[:,1:4]

kmeans = KMeans(2)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters

plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#! selecting the number of clusters

wcss=[] # list of wcss for each number of clusters (wcss for 1 cluster, wcss for 2 clusters...)

for i in range(1,7): # there are 6 observations
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_ # to get wcss
    wcss.append(wcss_iter)


#* the elbow method, showing why the optimal number of clusters is 3

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()