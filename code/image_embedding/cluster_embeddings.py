import os
import pickle
import numpy as np
from pathlib import Path
from som import SOM
from sklearn.cluster import KMeans

# Self-Organizing Map Parameters
SOM_LINES = 30
SOM_COLS = 30
SOM_ITERATIONS = 60
ALPHA = 0.4
SIGMA = 16.0

# Trains a Self-Organizing Map using the provided feature list and the globally defined parameters


def SelfOrganizingMap(feature_list):
    no_features = len(feature_list[0])
    print('Clustering', no_features, 'features')
    som = SOM(SOM_LINES, SOM_COLS, no_features, SOM_ITERATIONS, alpha=ALPHA, sigma=SIGMA)
    som.train(feature_list)

    image_grid = som.get_centroids()
    mapped = som.map_vects(feature_list)

    return mapped, image_grid


def K_Means(features):
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10).fit(features)
    kmeans_labels = kmeans.labels_
    return kmeans_labels


def cluster_embeddings(input_dir, embeddings_file, labels_file):
    # Read data
    embeddings = pickle.load(open(Path(input_dir / embeddings_file), 'rb'))
    labels = pickle.load(open(Path(input_dir / labels_file), 'rb'))

    output_dir = Path(__file__).parent / 'data'
    pickle.dump(embeddings, open(Path(output_dir / 'embeddings.p'), 'wb'))

    # KMeans Clustering
    kmeans_labels = K_Means(embeddings)

    output_dir = Path(__file__).parent / 'data'
    pickle.dump(kmeans_labels, open(Path(output_dir / 'kmeans_labels.p'), 'wb'))

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
        label_map = np.zeros((SOM_LINES, SOM_COLS), dtype=int)
        for vector, label in zip(mapped, labels_list):
            if target_label == label:
                label_map[vector[0]][vector[1]] += 1
        print(target_label)
        print('Label Map\n', '=======================')
        print(label_map)
        label_maps[target_label] = label_map

    output_dir = Path(__file__).parent / 'data'
    pickle.dump(mapped, open(Path(output_dir / 'mapped.p'), 'wb'))
    pickle.dump(labels_list, open(Path(output_dir / 'labels.p'), 'wb'))
    pickle.dump(label_maps, open(Path(output_dir / 'label_maps.p'), 'wb'))
    pickle.dump(image_grid, open(Path(output_dir / 'image_grid.p'), 'wb'))


z_dim = 10
curr_directory = Path(__file__).parent

curr_directory = curr_directory / "embedding_data"
next_folder = "z_" + str(z_dim)
input_dir = curr_directory / next_folder

embeddings_file = 'VAE_epoch031_z_tot.p'
labels_file = 'VAE_epoch031_id_tot.p'

print(Path(input_dir))

cluster_embeddings(Path(input_dir), embeddings_file, labels_file)
