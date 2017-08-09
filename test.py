# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:53:23 2017

@author: Admin
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin, euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
np.random.seed(0)
batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
print('the size of X: ', X.shape)
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(X)


k_means_cluster_centers = k_means.cluster_centers_
print ('the certers: ', k_means_cluster_centers)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
print ('the first 10 elements of k_means_labels: ', k_means_labels[:10])
print ('the size of k_means_labels: ', k_means_labels.shape)



