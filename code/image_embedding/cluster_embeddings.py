import os
import pickle
import numpy as np
from som import SOM
from sklearn.cluster import KMeans

# Self-Organizing Map Parameters
SOM_LINES = 20
SOM_COLS = 20
SOM_ITERATIONS = 40
ALPHA = 0.4
SIGMA = 16.0

# Trains a Self-Organizing Map using the provided feature list and the globally defined parameters
def SelfOrganizingMap(feature_list):
	no_features = len(feature_list[0])
	print ('Clustering', no_features, 'features')
	som = SOM(SOM_LINES, SOM_COLS, no_features, SOM_ITERATIONS, alpha = ALPHA, sigma = SIGMA)
	som.train(feature_list)

	image_grid = som.get_centroids()
	mapped = som.map_vects(feature_list)

	return mapped, image_grid

def K_Means(features):
	kmeans = KMeans(init = 'k-means++', n_clusters = 10, n_init = 10 ).fit(features)
	kmeans_labels = kmeans.labels_
	return kmeans_labels

def cluster_embeddings(input_dir, embeddings_file, labels_file):
	# Read data
	embeddings = pickle.load(open(input_dir + embeddings_file, 'rb'))
	labels = pickle.load(open(input_dir + labels_file, 'rb'))

	output_dir = './data/'
	pickle.dump(embeddings, open(output_dir + 'embeddings.p', 'wb'))

	# KMeans Clustering
	kmeans_labels = K_Means(embeddings)

	output_dir = './data/'
	pickle.dump(kmeans_labels, open(output_dir + 'kmeans_labels.p', 'wb'))

	# SOM Clustering
	feature_list = []
	for emb in list(embeddings):
		feature_list.append(list(emb))

	labels_list = []
	for label in list(labels):
		labels_list.append(str(np.argmax(label)))

	mapped, image_grid = SelfOrganizingMap(feature_list)

	label_maps = {}
	all_labels = [str(i) for i in range(10)]
	for target_label in all_labels:
		label_map = np.zeros((SOM_LINES, SOM_COLS), dtype = int)
		for vector, label in zip(mapped, labels_list):
			if target_label == label:
				label_map[vector[0]][vector[1]] += 1
		print (target_label)
		print ('Label Map\n', '=======================')
		print (label_map)
		label_maps[target_label] = label_map

	output_dir = './data/'
	pickle.dump(mapped, open(output_dir + 'mapped.p', 'wb'))
	pickle.dump(labels_list, open(output_dir + 'labels.p', 'wb'))
	pickle.dump(label_maps, open(output_dir + 'label_maps.p', 'wb'))
	pickle.dump(image_grid, open(output_dir + 'image_grid.p', 'wb'))

z_dim = 10
input_dir = './embedding_data/z_' + str(z_dim) + '/'
embeddings_file = 'VAE_epoch031_z_tot.p'
labels_file = 'VAE_epoch031_id_tot.p'

cluster_embeddings(input_dir, embeddings_file, labels_file)
