import numpy as np
import math
import sys

class KMeans(object):
	def __init__(self, n_clusters=8, n_init=10, max_iter=300, random_state=None):
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.max_iter = max_iter
		self.random_state = random_state

	"""
	X: [[x1, y1], [x2, y2], [x3, y3]]
	"""
	def fit(self, X):
		X = np.array(X)

		#create random instance with seed
		rgen = np.random.RandomState(self.random_state)

		#find min, max of first, second column (x, y axes)
		minX, maxX = X[:, 0].min(), X[:, 0].max()
		minY, maxY = X[:, 1].min(), X[:, 1].max()

		#initiate variables
		self.labels_ = [0] * X.shape[0]
		self.inertia_ = sys.float_info.max

		#randomize centroids n_init times
		for _ in range(self.n_init):
			#create random centroids
			k = [np.random.uniform(low=minX, high=maxX, size=self.n_clusters),
				np.random.uniform(low=minY, high=maxY, size=self.n_clusters)]
			k = np.array(k).T

			#create labels
			labels = np.array([0] * X.shape[0])

			#clustering max_iter times
			for epoch in range(self.max_iter):
				clusters_changed = False

				#find minimum distances between points and centroids
				for x_index, point in enumerate(X):
					distance = sys.float_info.max
					for k_index, centroid in enumerate(k):
						dist = self.calc_distance(point, centroid)
						if dist < distance:
							distance = dist
							labels[x_index] = k_index

				#update centroids (k), cluster_changed
				for k_index, centroid in enumerate(k):
					old_centroid = centroid.copy()
					cluster = X[labels == k_index]
					if cluster.any():
						centroid[0] = np.average(cluster[:,0])
						centroid[1] = np.average(cluster[:,1])
						if old_centroid[0] != centroid[0] or old_centroid[1] != centroid[1]:
							clusters_changed = True

				#cluster_changed == False -> break max_iter
				if not clusters_changed:
					break

			#find the best result by finding the minimum total variation
			total_variation = self.calc_total_variation(X, k, labels)
			if total_variation < self.inertia_:
				self.inertia_ = total_variation
				self.labels_ = labels
		return self

	"""
	X: [[x1, y1], [x2, y2], [x3, y3]]
	k: [[x1, y1], [x2, y2], [x3, y3]]
	labels: [0, 1, 2]
	"""
	def calc_total_variation(self, X, k, labels):
		total_variation = 0.
		for k_index, centroid in enumerate(k):
			clustered_X = X[labels == k_index]
			for point in clustered_X:
				x = point[0]-centroid[0]
				y = point[1]-centroid[1]
				total_variation += x*x + y*y
		return total_variation

	"""
	calculates distance from point a to point b
	a: [x1, y1]
	b: [x2, y2]
	"""
	def calc_distance(self, a, b):
		x = b[0]-a[0]
		y = b[1]-a[1]
		return math.sqrt(x*x + y*y)