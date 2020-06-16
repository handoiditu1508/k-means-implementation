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
		rgen = np.random.RandomState(self.random_state)
		minX, maxX = X[:, 0].min(), X[:, 0].max()
		minY, maxY = X[:, 1].min(), X[:, 1].max()
		self.labels_ = [0] * X.shape[0]
		self.inertia_ = sys.float_info.max
		for _ in range(self.n_init):
			k = np.concatenate((np.random.uniform(low=minX, high=maxX, size=self.n_clusters),
								np.random.uniform(low=minY, high=maxY, size=self.n_clusters)),
								axis = 1)
			labels = [0] * X.shape[0]
			for _2 in range(self.max_iter):
				clusters_changed = False
				for x_index, point in enumerate(X):
					distance = sys.float_info.max
					for k_index, centroid in enumerate(k):
						dist = calc_distance(point, centroid)
						if dist < distance:
							distance = dist
							labels[x_index] = k_index
				#update centroids (k), cluster_changed
				#cluster_changed == False -> break
			total_variation = calc_total_variation(X, k, labels)
			if total_variation < self.inertia_:
				self.inertia_ = total_variation
				self.labels_ = labels

	"""
	X: [[x1, y1], [x2, y2], [x3, y3]]
	k: [[x1, y1], [x2, y2], [x3, y3]]
	labels: [0, 1, 2]
	"""
	def calc_total_variation(X, k, labels)
		distances = []
		for idx, centroid in enumerate(k):
			clustered_X = X[labels == idx]
			for point in clustered_X:
				distances.append(calc_distance(centroid, point))
		return np.dot[distances, distances]

	"""
	calculates distance from point a to point b
	a: [x1, y1]
	b: [x2, y2]
	"""
	def calc_distance(a, b):
		x = b[0]-a[0]
		y = b[1]-a[1]
		return math.sqrt(x*x + y*y)