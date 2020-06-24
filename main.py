import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from kmeans import KMeans
from sklearn.datasets import make_blobs

features, target = make_blobs(n_samples = 100,n_features = 2,centers = 5,cluster_std = 0.5,random_state = 10)

kmeans = KMeans(n_clusters = 5, n_init = 10)
kmeans = kmeans.fit(features)

plt.scatter(features[:,0], features[:,1], c = kmeans.labels_)
plt.show()