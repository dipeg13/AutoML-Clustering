import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from cdbw import CDbw
from ClustersFeatures import *

np.random.seed(666)

def clusterization(data, labels):
    clusters = {}
    for lab in np.unique(labels):
        clusters[lab] = []
    for i in range(data.shape[0]):
        clusters[labels[i]].append(data[i,:])
    return clusters

#From https://github.com/iphysresearch/S_Dbw_validity_index
class S_Dbw():
    def __init__(self,data,data_cluster,cluster_centroids_):
        """
        data --> raw data
        data_cluster --> The category that represents each piece of data(the number of category should begin 0)
        cluster_centroids_ --> the center_id of each cluster's center
        """
        self.data = data
        self.data_cluster = data_cluster
        self.cluster_centroids_ = cluster_centroids_

        self.k = cluster_centroids_.shape[0]
        self.stdev = 0
        for i in range(self.k):
            std_matrix_i = np.std(data[self.data_cluster == i],axis=0)
            self.stdev += np.sqrt(np.dot(std_matrix_i.T,std_matrix_i))
        self.stdev = np.sqrt(self.stdev)/self.k 


    def density(self,density_list=[]):
        """
        compute the density of one or two cluster(depend on density_list)
        """
        density = 0
        if len(density_list) == 2:
            center_v = (self.cluster_centroids_[density_list[0]] +self.cluster_centroids_[density_list[1]])/2
        else:
            center_v = self.cluster_centroids_[density_list[0]]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:    # np.linalg.norm(order=2)
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density


    def Dens_bw(self):
        density_list = []
        result = 0
        #density_list
        for i in range(self.k):
            density_list.append(self.density(density_list=[i]))
        for i in range(self.k):
            for j in range(self.k):
                if i==j:
                    continue
                result += self.density([i,j])/max(density_list[i],density_list[j])
        return result/(self.k*(self.k-1))

    def Scat(self):
        sigma_s = np.std(self.data,axis=0)
        sigma_s_2norm = np.sqrt(np.dot(sigma_s.T,sigma_s))

        sum_sigma_2norm = 0
        for i in range(self.k):
            matrix_data_i = self.data[self.data_cluster == i]
            sigma_i = np.std(matrix_data_i,axis=0)
            sum_sigma_2norm += np.sqrt(np.dot(sigma_i.T,sigma_i))
        return sum_sigma_2norm/(sigma_s_2norm*self.k)


    def S_Dbw_result(self):
        """
        compute the final result
        """
        return self.Dens_bw()+self.Scat()

def SDbw(algorithm, data):
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    
    algorithm_cluster_centers = algorithm.cluster_centers_
    algorithm_labels = pairwise_distances_argmin(data, algorithm_cluster_centers)

    KS = S_Dbw(data, algorithm_labels, algorithm_cluster_centers)
    return KS.S_Dbw_result()

def cIndex(clusters):
    clusterLabels = []
    clusterObjects = []
    for key in clusters.keys():
        for item in clusters[key]:
            clusterLabels.append(key)
            clusterObjects.append(item)
    from c_index import calc_c_index
    return calc_c_index(np.asarray(clusterObjects), clusterLabels)

def CDBW(clusters):
    clusterLabels = []
    clusterObjects = []
    for key in clusters.keys():
        for item in clusters[key]:
            clusterLabels.append(key)
            clusterObjects.append(item)
    
    return CDbw(np.asarray(clusterObjects), clusterLabels, metric='euclidean')

def indices(clusters, algorithm, data):
    clusterLabels = []
    clusterObjects = []
    for key in clusters.keys():
        for item in clusters[key]:
            clusterLabels.append(key)
            clusterObjects.append(item)
    from sklearn.datasets import load_digits
    import pandas as pd
    digits = load_digits()
    pd_df = pd.DataFrame(clusterObjects)
    pd_df['target'] = clusterLabels
    CC=ClustersCharacteristics(pd_df,label_target="target")

    indi = CC.general_info()
    dic = {}
    dic['Silhouette'] = indi.iloc[2][0]
    dic['Dunn'] = indi.iloc[3][0]
    dic['C-Index'] = indi.iloc[13][0]
    dic['Calinski-Harabasz'] = indi.iloc[5][0]
    dic['Davies Bouldin'] = indi.iloc[16][0]
    #dic['SDbw'] = SDbw(algorithm, data)
    #dic['CDBW'] = CDBW(clusters)
    #dic['Tau'] = 
    dic['Ratkowsky Lance'] = indi.iloc[6][0]
    dic['McClain Rao'] = indi.iloc[18][0]
    return dic

    
#--------------------------------------------------------------------------------
from sklearn.cluster import AffinityPropagation as affinityProp
from sklearn.cluster import AgglomerativeClustering as agglomerative
from sklearn.cluster import Birch as birch
from sklearn.cluster import DBSCAN as dbscan
from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MeanShift as meanshift
from sklearn.cluster import OPTICS as optics
from sklearn.cluster import SpectralClustering as spectral
#--------------------------------------------------------------------------------

searchSpace = {'affinity' : {'damping': [i * 0.1 for i in range(5,10)]},
               'agglomerative' : {'n_clusters' : [i for i in range(2, 31)],
                                  'affinity' : ['euclidean', 'l', 'l2', 'manhattan', 'cosine']},
               'birch' : {'threshold' : [i * 0.1 for i in range(2,8)],
                          'n_clusters': [i for i in range(2, 31)]},
               'dbscan' : {'eps' : [i * 0.1 for i in range(1,8)],
                           'min_samples' : [i for i in range(3,8)]},
               'kmeans' : {'n_clusters' : [i for i in range(2,31)]},
               'meanshift' : {'bandwidth' : None},
               'optics' : {'min_samples' : [i for i in range(3,8)],
                           'cluster_method' : ['xi', 'dbscan']},
               'spectral' : {'n_clusters' : [i for i in range(2,31)],
                             'gamma' : [i*0.1 for i in range(10,15)]}}

def AffinityClustering(data, damping):
    algorithm = affinityProp(damping=damping).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def AgglomerativeClustering_(data, n_clusters, affinity):
    algorithm = agglomerative(n_clusters=n_clusters, affinity=affinity).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def BirchClustering(data, threshold, n_clusters):
    algorithm = birch(threshold=threshold, n_clusters=n_clusters).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def DBSCANClustering(data, eps, min_samples):
    algorithm = dbscan(eps=eps, min_samples=min_samples).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def KMeansClustering(data, n_clusters):
    algorithm = kmeans(n_clusters=n_clusters).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def MeanShiftClustering(data, bandwidth):
    algorithm = meanshift(bandwidth=bandwidth).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def OpticsClustering(data, min_samples, cluster_method):
    algorithm = optics(min_samples=min_samples, cluster_method=cluster_method).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)
def SpectralClustering(data, n_clusters, gamma):
    algorithm = spectral(n_clusters=n_clusters, gamma=gamma).fit(data)
    clusters = clusterization(np.asarray(data), algorithm.labels_)
    return indices(clusters, algorithm, data)

import os

datasets = {}
first = True
for root, dirs, files in os.walk(r'TransformedDatasets'):
    for file in files:
        df = pd.read_csv(r'TransformedDatasets\\'+file)
        print(r'TransformedDatasets\\'+file)
        datasets[file] = []
        if file != 'dataset 1' and file != 'dataset 10' and file != 'dataset 100':
            for algo in searchSpace.keys():
                if algo == 'kmeans':
                    print('Kmeans started')
                    for n_clusters in searchSpace[algo]['n_clusters']:
                        results = KMeansClustering(df, n_clusters)
                        datasets[file].append(('KMeans', {'n_clusters':n_clusters}, results))
                        print('n_clusters-'+str(n_clusters))
                        print(results)
                    print(file , algo , 'ended')
                
                if algo == 'meanshift':
                    print('Mean Shift started')
                    results = MeanShiftClustering(df, bandwidth=None)
                    datasets[file].append(('MeanShiftClustering', {'bandwidth':None}, results))
                    print(results)
                    print(file , algo , 'ended')
            """
            if algo == 'affinity':
                print('Affinity started')
                for damping in searchSpace[algo]['damping']:
                    results = AffinityClustering(df, damping)
                    datasets[file].append(('AffinityClustering', {'damping':damping}, results))
                    print('damping-'+str(damping))
                    print(results)
                print(file , algo , 'ended')
            if algo == 'agglomerative':
                print('Agglomerative started')
                for n_clusters in searchSpace[algo]['n_clusters']:
                    for affinity in searchSpace[algo]['affinity']:
                        results = AgglomerativeClustering_(df, n_clusters, affinity)
                        dataset[file].append(('AgglomerativeClustering', {'n_clusters':n_clusters, 'affinity':affinity}, results))
                        print('n_clusters-'+str(n_clusters)+', affinity-'+affinity)
                        print(results)
                print(file , algo , 'ended')
            if algo == 'birch':
                print('Birch started')
                for threshold in searchSpace[algo]['threshold']:
                    for n_clusters in searchSpace[algo]['n_clusters']:
                        results = BirchClustering(df, threshold, n_clusters)
                        dataset[file].append(('BirchClustering', {'threshold':threshold, 'n_clusters':n_clusters}, results))
                        print('threshold-'+str(threshold)+', n_clusters-'+str(n_clusters))
                        print(results)
                print(file , algo , 'ended')
            if algo == 'dbscan':
                print('DBSCAN started')
                for eps in searchSpace[algo]['eps']:
                    for min_samples in searchSpace[algo]['min_samples']:
                        results = DBSCANClustering(df, eps, min_samples)
                        dataset[file].append(('DBSCANClustering', {'eps':eps, 'min_samples':min_samples}, results))
                        print('eps-'+str(eps)+', mean_samples-'+str(mean_samples))
                        print(results)
                print(file , algo , 'ended')
            """
            """
            if algo == 'kmeans':
                print('Kmeans started')
                for n_clusters in searchSpace[algo]['n_clusters']:
                    results = KMeansClustering(df, n_clusters)
                    datasets[file].append(('KMeans', {'n_clusters':n_clusters}, results))
                    print('n_clusters-'+str(n_clusters))
                    print(results)
                print(file , algo , 'ended')
            
            if algo == 'meanshift':
                print('Mean Shift started')
                results = MeanShiftClustering(df, bandwidth=None)
                datasets[file].append(('MeanShiftClustering', {'bandwidth':None}, results))
                print(results)
                print(file , algo , 'ended')
            """
            """
            
            if algo == 'optics':
                print('Optics started')
                for min_samples in searchSpace[algo]['min_samples']:
                    for cluster_method in searchSpace[algo]['cluster_method']:
                        results = OpticsClustering(df, min_samples, cluster_method)
                        datasets[file].append(('OpticsClustering', {'min_samples':min_samples, 'cluster_method':cluster_method}, results))
                        print('min_samples-'+str(min_samples)+', cluster_method-'+cluster_method)
                        print(results)
                print(file , algo , 'ended')
            
            if algo == 'spectral':
                print('Spectral started')
                for n_clusters in searchSpace[algo]['n_clusters']:
                    for gamma in searchSpace[algo]['gamma']:
                        results = SpectralClustering(df, n_clusters, gamma)
                        datasets[file].append(('SpectralClustering', {'n_clusters':n_clusters, 'gamma':gamma}, results))
                        print('n_clusters-'+str(n_clusters)+', gamma-'+str(gamma))
                        print(results)
                print(file , algo , 'ended')
            """

import pickle
a_file = open("data.pkl", "wb")

pickle.dump(datasets, a_file)

a_file.close()
 
