import pickle
import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
from pathlib import Path

input_dir = Path(__file__).parent / 'data'

kmeans_labels = pickle.load(open(Path(input_dir / 'kmeans_labels.p'), 'rb'))
mapped = pickle.load(open(Path(input_dir / 'mapped.p'), 'rb'))
embeddings = pickle.load(open(Path(input_dir / 'embeddings.p'), 'rb'))
labels = pickle.load(open(Path(input_dir / 'labels.p'), 'rb'))


def get_classification_map():
    num_classes = 10
    max_row = 0
    max_col = 0
    for location in mapped:
        max_row = max(max_row, location[0])
        max_col = max(max_col, location[1])

    activation_map = np.zeros((max_row+1, max_col+1, num_classes), dtype=np.uint8)
    for label, location in zip(labels, mapped):
        activation_map[location[0], location[1], int(label)] += 1
    classification_map = np.argmax(activation_map, axis=-1)
    print(classification_map)
    return classification_map


classification_map = get_classification_map()
som_labels = []
for location in mapped:
    som_labels.append(classification_map[location[0], location[1]])


# KMeans
km_silhouette_coef = silhouette_score(embeddings, kmeans_labels)
km_homo = homogeneity_score(labels, kmeans_labels)
km_comp = completeness_score(labels, kmeans_labels)
print('KM silhouette score', km_silhouette_coef)
print('KM homogeneity', km_homo)
print('KM completeness', km_comp)

# Self Organizing Map
som_silhouette_coef = silhouette_score(mapped, labels)
som_homo = homogeneity_score(labels, som_labels)
som_comp = completeness_score(labels, som_labels)
print('SOM silhouette score', som_silhouette_coef)
print('SOM homogeneity', som_homo)
print('SOM completeness', som_comp)
