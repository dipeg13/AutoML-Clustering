import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import zscore


def distanceVector(data):
    distanceVector = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            distanceVector.append(distance(data[i,:].reshape(1,-1), data[j,:].reshape(1,-1)))
    distanceVector = np.asarray(distanceVector)
    return distanceVector/np.linalg.norm(distanceVector)

#The method below returns the meta-features presented in Ferrari et al. paper
def distanceBasedExtraction(data):
    dVector = distanceVector(data)
    distanceFeatures = np.zeros(shape=(19,1))
    distanceFeatures[0] = np.mean(dVector)
    distanceFeatures[1] = np.var(dVector)
    distanceFeatures[2] = np.std(dVector)
    distanceFeatures[3] = skew(dVector)
    distanceFeatures[4] = kurtosis(dVector)
    distanceFeatures[5] = np.count_nonzero(dVector <= 0.1) / dVector.shape[1]
    distanceFeatures[6] = np.count_nonzero(np.logical_and(0.1 < dVector, dVector <= 0.2)) / dVector.shape[1]
    distanceFeatures[7] = np.count_nonzero(np.logical_and(0.2 < dVector, dVector <= 0.3)) / dVector.shape[1]
    distanceFeatures[8] = np.count_nonzero(np.logical_and(0.3 < dVector, dVector <= 0.4)) / dVector.shape[1]
    distanceFeatures[9] = np.count_nonzero(np.logical_and(0.4 < dVector, dVector <= 0.5)) / dVector.shape[1]
    distanceFeatures[10] = np.count_nonzero(np.logical_and(0.5 < dVector, dVector <= 0.6)) / dVector.shape[1]
    distanceFeatures[11] = np.count_nonzero(np.logical_and(0.6 < dVector, dVector <= 0.7)) / dVector.shape[1]
    distanceFeatures[12] = np.count_nonzero(np.logical_and(0.7 < dVector, dVector <= 0.8)) / dVector.shape[1]
    distanceFeatures[13] = np.count_nonzero(np.logical_and(0.8 < dVector, dVector <= 0.9)) / dVector.shape[1]
    distanceFeatures[14] = np.count_nonzero(np.logical_and(0.9 < dVector, dVector <= 1)) / dVector.shape[1]
    distanceFeatures[15] = np.count_nonzero(np.logical_and(0 <= zscore(dVector), zscore(dVector) < 1)) / dVector.shape[1]
    distanceFeatures[16] = np.count_nonzero(np.logical_and(1 <= zscore(dVector), zscore(dVector) < 2)) / dVector.shape[1]
    distanceFeatures[17] = np.count_nonzero(np.logical_and(2 <= zscore(dVector), zscore(dVector) < 3)) / dVector.shape[1]
    distanceFeatures[18] = np.count_nonzero(3 <= zscore(dVector)) / dVector.shape[1]
    return distanceFeatures

from sklearn.datasets import load_iris

data = load_iris().data

#mf = distanceBasedExtraction(data)

#The method below returns the cluster in dictionary format
def clusterization(data, labels):
    clusters = {}
    for lab in np.unique(labels):
        clusters[lab] = []
    for i in range(data.shape[0]):
        clusters[labels[i]].append(data[i,:])
    return clusters

def silhouette(clusters):
    pass

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(data)

clusters = clusterization(data, km.labels_)
