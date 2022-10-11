import json
import os.path
import numpy as np
import pandas as pd
import tqdm
from autoclust.utils.mspecs import optimize_alg_space
from autoclust.validation import compute_internal_cvi, compute_external_cvi
from sklearn.cluster import Birch, DBSCAN
import random
import time

"""
Random search for datasets without ground truth
"""
def random_search_internal(df, max_trials=300):
    paramsBirch = {'threshold' : [0.001, 1], 'n_clusters' : [2, 10]}
    paramsDBSCAN = {'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'eps' : [0.001, 1],
                    'metric' : ['euclidean', 'cosine'],
                    'min_samples' : [2, 10],
                    'p' : [1, 2],
                    'n_jobs' : [-2]
                    }
    dataBirch = pd.DataFrame()
    dataDBSCAN = pd.DataFrame()
    max_eval = 50
    for i in range(max_trials):

        BirchThreshold = np.random.uniform(paramsBirch['threshold'][0], paramsBirch['threshold'][1])
        BirchNClusters = np.random.randint(paramsBirch['n_clusters'][0], paramsBirch['n_clusters'][1]+1)
        paramsB = {'threshold' : BirchThreshold, 'n_clusters' : BirchNClusters}
        #paramsB = {'n_clusters' : BirchNClusters}
        print(paramsB)
        try:
            start_time = time.time()
            alg = Birch(threshold=BirchThreshold, n_clusters=BirchNClusters)
            #alg = Birch(n_clusters=BirchNClusters)
            
            fitted_model = alg.fit(df)
            labels = fitted_model.labels_
            evaluation_dict = compute_internal_cvi(df, cvi_engine="r", labels=fitted_model.labels_)
            evaluation_dict["Algorithm"] = 'Birch'
            evaluation_dict["parameters"] = paramsB
            evaluation_dict["exec_time_"] = time.time() - start_time
            dataBirch = dataBirch.append(evaluation_dict, ignore_index=True)
            #data = data.concat(pd.DataFrame(data=[list(evaluation_dict.values)], columns=[list(evaluation_dict.keys())]), ignore_index=True)
        except Exception as e:
            print(e)
        
        DBSCANAlgorithm = np.random.choice(np.asarray(paramsDBSCAN['algorithm']))
        DBSCANEps = np.random.uniform(paramsDBSCAN['eps'][0], paramsDBSCAN['eps'][1])
        if DBSCANAlgorithm == 'kd_tree':
            DBSCANMetric = 'euclidean'
        elif DBSCANAlgorithm == 'ball_tree':
            DBSCANMetric = 'euclidean'
        else:
            DBSCANMetric = random.choice(paramsDBSCAN['metric'])
        DBSCANMinSamples = np.random.randint(paramsDBSCAN['min_samples'][0], paramsDBSCAN['min_samples'][1]+1)
        DBSCANP = np.random.uniform(paramsDBSCAN['p'][0], paramsDBSCAN['p'][1]+0.0001)
        DBSCANNJobs = -2
        
        paramsD = {'algorithm' : DBSCANAlgorithm, 'eps' : DBSCANEps, 'metric' : DBSCANMetric, 'min_samples' : DBSCANMinSamples, 'p' : DBSCANP, 'n_jobs' : DBSCANNJobs}
        
        print(paramsD)
        try:
            start_time = time.time()
            alg = DBSCAN(algorithm=DBSCANAlgorithm, eps=DBSCANEps, metric=DBSCANMetric, min_samples=DBSCANMinSamples, p=DBSCANP, n_jobs=DBSCANNJobs)
            fitted_model = alg.fit(df)
            labels = fitted_model.labels_
            evaluation_dict = compute_internal_cvi(df, cvi_engine="r", labels=fitted_model.labels_)
            evaluation_dict["Algorithm"] = 'DBSCAN'
            evaluation_dict["parameters"] = paramsD
            evaluation_dict["exec_time_"] = time.time() - start_time
            dataDBSCAN = dataDBSCAN.append(evaluation_dict, ignore_index=True)

        except Exception as e:
            print(e)
    
    results = pd.DataFrame()
    results = results.append(dataBirch).append(dataDBSCAN)
    return results

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

"""
Random Search for datasets with ground truth
"""
def random_search_external(df, target, max_trials=300):
    paramsBirch = {'threshold' : [0.001, 1], 'n_clusters' : [2, 10]}
    paramsDBSCAN = {'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'eps' : [0.001, 1],
                    'metric' : ['euclidean', 'cosine'],
                    'min_samples' : [2, 10],
                    'p' : [1, 2],
                    'n_jobs' : [-2]
                    }
    dataBirch = pd.DataFrame()
    dataDBSCAN = pd.DataFrame()
    max_eval = 50
    for i in range(max_trials):

        BirchThreshold = np.random.uniform(paramsBirch['threshold'][0], paramsBirch['threshold'][1])
        BirchNClusters = np.random.randint(paramsBirch['n_clusters'][0], paramsBirch['n_clusters'][1]+1)
        paramsB = {'threshold' : BirchThreshold, 'n_clusters' : BirchNClusters}
        #paramsB = {'n_clusters' : BirchNClusters}
        print(paramsB)
        try:
            start_time = time.time()
            alg = Birch(threshold=BirchThreshold, n_clusters=BirchNClusters)
            #alg = Birch(n_clusters=BirchNClusters)
            
            fitted_model = alg.fit(df)
            labels = fitted_model.labels_
            evaluation_dict = compute_internal_cvi(df, cvi_engine="r", labels=fitted_model.labels_)
            external = compute_external_cvi(labels, target)
            evaluation_dict.update(external)
            evaluation_dict["Algorithm"] = 'Birch'
            evaluation_dict["parameters"] = paramsB
            evaluation_dict["exec_time_"] = time.time() - start_time
            evaluation_dict["fitted_labels"] = list(labels)
            dataBirch = dataBirch.append(evaluation_dict, ignore_index=True)
            #data = data.concat(pd.DataFrame(data=[list(evaluation_dict.values)], columns=[list(evaluation_dict.keys())]), ignore_index=True)
        except Exception as e:
            print(e)
        
        DBSCANAlgorithm = np.random.choice(np.asarray(paramsDBSCAN['algorithm']))
        DBSCANEps = np.random.uniform(paramsDBSCAN['eps'][0], paramsDBSCAN['eps'][1])
        if DBSCANAlgorithm == 'kd_tree':
            DBSCANMetric = 'euclidean'
        elif DBSCANAlgorithm == 'ball_tree':
            DBSCANMetric = 'euclidean'
        else:
            DBSCANMetric = random.choice(paramsDBSCAN['metric'])
        DBSCANMinSamples = np.random.randint(paramsDBSCAN['min_samples'][0], paramsDBSCAN['min_samples'][1]+1)
        DBSCANP = np.random.uniform(paramsDBSCAN['p'][0], paramsDBSCAN['p'][1]+0.0001)
        DBSCANNJobs = -2
        
        paramsD = {'algorithm' : DBSCANAlgorithm, 'eps' : DBSCANEps, 'metric' : DBSCANMetric, 'min_samples' : DBSCANMinSamples, 'p' : DBSCANP, 'n_jobs' : DBSCANNJobs}
        
        print(paramsD)
        try:
            start_time = time.time()
            alg = DBSCAN(algorithm=DBSCANAlgorithm, eps=DBSCANEps, metric=DBSCANMetric, min_samples=DBSCANMinSamples, p=DBSCANP, n_jobs=DBSCANNJobs)
            fitted_model = alg.fit(df)
            labels = fitted_model.labels_
            evaluation_dict = compute_internal_cvi(df, cvi_engine="r", labels=fitted_model.labels_)
            external = compute_external_cvi(labels, target)
            evaluation_dict.update(external)
            evaluation_dict["Algorithm"] = 'DBSCAN'
            evaluation_dict["parameters"] = paramsD
            evaluation_dict["exec_time_"] = time.time() - start_time
            evaluation_dict["fitted_labels"] = list(labels)
            dataDBSCAN = dataDBSCAN.append(evaluation_dict, ignore_index=True)

        except Exception as e:
            print(e)
    
    results = pd.DataFrame()
    results = results.append(dataBirch).append(dataDBSCAN)
    return results



#df = pd.read_csv(r'C:\Users\user\Desktop\Δημήτρης\ExhaustiveSearch-Clustering-main\Datasets\openml\ESL\clean_data.csv', header=None)
#target = np.genfromtxt(r'C:\Users\user\Desktop\Δημήτρης\ExhaustiveSearch-Clustering-main\Datasets\openml\ESL\raw_data\target.csv')





#results = random_search_external(df, target, 20)
