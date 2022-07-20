#import sklearn.metrics.silhouette_score as silhouette
#Dunn
#C-Index
#import sklearn.metrics.calinski_harabasz_score as Calinski_Harabasz
#import sklearn.metrics.davies_bouldin_score as Davies_Bouldin
#SDbw
#CDBW
#Tau
#Ratkowsky Lance
#McClain Rao
"""
import sklearn.cluster.AffinityPropagation as affinity
import sklearn.cluster.AgglomerativeClustering as agglomerative
import sklearn.cluster.Birch as birch
import sklearn.cluster.DBSCAN as dbscan
import sklearn.cluster.KMeans as kmeans
import sklearn.cluster.MeanShift as meanshift
import sklearn.cluster.OPTICS as optics
import sklearn.cluster.SpectralClustering as spectral

searchSpace = {'affinity' : {'damping': [i * 0.1 for i in range(5,0)]},
               'agglomerative' : {'n_clusters' : [i for i in range(2, 31)],
                                  'affinity' : ['euclidean', 'l', 'l2', 'manhattan', 'cosine']},
               'birch' : {'threshold' : [i * 0.1 for i in range(2,8)],
                          'n_clusters': [i for i in range(2, 31)]},
               'dbscan' : {'eps' : [i * 0.1 for i in range(1,8)],
                           'mean_samples' : [i for i in range(3,8)]},
               'kmeans' : {'n_clusters' : [i for i in range(2,31)]},
               'meanshift' : {'bandwidth' : None},
               'optics' : {'min_samples' : [i for i in range(3,8)],
                           'cluster_method' : ['xi', 'dbscan']},
               'spectral' : {'n_clusters' : [i for i in range(2,31)],
                             'gamma' : [i*0.1 for i in range(5,15)]}}

"""
import numpy as np
import matplotlib.pyplot as plt
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
    #https://en.wikipedia.org/wiki/Silhouette_(clustering)
    def a(dp, label):
        global clusters
        summ = 0
        for i in clusters[label]:
            if i != dp:
                summ += np.linalg.norm(dp -i)
        return summ / (len(clusters[label]) - 1)
    
    def b(dp, label):
        global clusters
        #Below we create a dictionary {label : min} to catch easier the minimum cluster
        cluster_min = {}
        cluster_labels = list(clusters.keys())
        counter = 0
        for lab in cluster_labels:
            if lab != label:
                temp_sum = 0
                for t in clusters[lab]:
                    temp_sum += np.linalg.norm(dp - t)
                cluster_min[lab] = temp_sum / len(clusters[lab])
        cl_list = list(cluster_min.keys())
        minimum_cluster = cluster_min[cl_list[0]]
        for i in cl_list:
            if minimum_cluster <= cluster_min[i]:
                minimum_cluster = cluster_min[i]
        return minimum_cluster

    total_dataPoints = 0
    for i in clusters.keys():
        total_dataPoints += len(clusters[i])
        
    s = np.zeros(total_dataPoints)

    counter = 0
    for l in clusters.keys():
        for inst in clusters[l]:
            if len(clusters[l]) == 1:
                s[counter] = 0
            else:
                s[counter] = (b(inst, l) - a(inst, l)) / max(a(inst, l), b(inst, l))
    return np.mean(s)

def DaviesBouldin(clusters):
    def cent_maker(data):
        counter = 1
        cent = np.asarray(data[0])
        for i in range(1,len(data)):
            cent += np.asarray(data[i])
            counter += 1
        return cent / counter

    def Diameter(data):
        centroid = cent_maker(data)
        counter = 0
        summ = 0
        for i in range(len(data)):
            summ += np.linalg.norm(np.asarray(data[i]) - np.asarray(centroid))
            counter += 1
        return summ / counter

    def maxFrac(i_cluster, clusters):
        maximum = -1
        for i in clusters.keys():
            if i != i_cluster:
                numerator = Diameter(clusters[i]) + Diameter(clusters[i_cluster])
                denumerator = np.linalg.norm(cent_maker(clusters[i]) - cent_maker(clusters[i_cluster]))
                if maximum <= numerator / denumerator:
                    maximum = numerator / denumerator
        return maximum

    cl_labels = list(clusters.keys())
    summ = 0
    for i in cl_labels:
        summ += maxFrac(i, clusters)
    return summ / len(cl_labels)


    

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(data)

clusters = clusterization(data, km.labels_)
"""
#print(silhouette(clusters))
    
from sklearn.metrics import davies_bouldin_score as bd
bdsk = []
bdmine = []
for i in range(2, 50):
    km = KMeans(n_clusters=i).fit(data)
    clusters = clusterization(data, km.labels_)
    bdsk.append(bd(data, km.labels_))
    bdmine.append(DaviesBouldin(clusters))
m = [i for i in range(2,50)]
plt.plot(m, bdsk, label='sk-learn')
plt.plot(m, bdmine, label='mine')
plt.legend()
plt.show()
"""
