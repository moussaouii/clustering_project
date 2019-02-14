import numpy as np
from sklearn.cluster import KMeans

def clustering_KMeans(data,n_clusters,random_state):
    # intiate kmeans model
    kmeans = KMeans(n_clusters=n_clusters,random_state=random_state)
    # Fitting the input data
    kmeans_model = kmeans.fit(data)
    # Getting the cluster labels
    labels = kmeans.predict(data)
    # Expand the shape of an array
    labels = np.expand_dims(labels, axis=1)
    # return the results
    return np.hstack((data,labels)), kmeans
