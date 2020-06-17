import os
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from kmeans import KMeans
import math
from sklearn.datasets import make_blobs

clear = lambda: os.system('cls')
clear()

#calculate distance between point and line in 2d
def calc_distance(x1, y1, a, b, c):
	return abs(a*x1 + b*y1 + c) / math.sqrt(a*a + b*b)

#create test data
data, target = make_blobs(n_samples = 100,n_features = 2,centers = 5,cluster_std = 0.5,random_state = 10)

#list of each KMeans's summation of
#distances from data points to centroid
#or within-cluster sum of squares (WCSS)
totalVariations = []
#store KMeans object for each K value
kmeansStorage = []

#list of k values
Ks = range(1, 8)
for i in Ks:
	kmeans = KMeans(n_clusters=i, n_init=10)
	kmeans = kmeans.fit(data)
	totalVariations.append(kmeans.inertia_)
	kmeansStorage.append(kmeans)

#ax + by + c = 0
#(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
x1 = Ks[0]
y1 = totalVariations[0]
x2 = Ks[-1]
y2 = totalVariations[-1]
a = y2 - y1
b = x1 - x2
c = x2*y1 - x1*y2

#find final K value index
maxDistance = 0.
finalIndex = 0
for i in range(len(Ks)):
	d = calc_distance(Ks[i], totalVariations[i], a, b, c)
	print(str(Ks[i])+": "+str(d))
	if maxDistance < d:
		maxDistance = d
		finalIndex = i

#final K value
print(Ks[finalIndex])

#show K values and it's total variation
plt.plot(Ks, totalVariations, c="r")
plt.plot([x1, x2], [y1, y2], c="b")
plt.xlabel("K values")
plt.ylabel("Total variations")
plt.show()

#show clusters
plt.scatter(data[:,0], data[:,1], c=kmeansStorage[finalIndex].labels_)
plt.show()

#show true clusters
plt.scatter(data[:,0], data[:,1], c=target)
plt.show()