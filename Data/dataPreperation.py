import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer as knn

def preprocessing(df):
    #The method below drops all columns with a single value
    def dropSingleValues(df):
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col,inplace=True,axis=1)
        return df
    #The method below replaces infinity values with nans
    #def replaceInfty(df):
    #    return df.replace([np.inf, -np.inf], 'infty', inplace=True)
    #The method below drops all columns with 30% of missing values
    def dropMissingThirty(df):
        for col in df.columns:
            if df[col].isnull().sum() / len(df[col]) > .3:
                df.drop(col, inplace=True, axis=1)
        return df
    #The method below transforms "Object' type columns to numerical
    def numerizor(df):
        for col in df.columns:
            if df[col].dtypes == 'O':
                df[col] = LabelEncoder().fit_transform(df[col])
        return df
    #The method below conducts [0,1] min-max scaling
    def minMax(df):
        colNames = list(df.columns)
        return pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(df), columns=colNames)
    #The method below conducts 10-nn imputation for replacing missing values
    def knnMissingValues(df, k=10):
        colNames = list(df.columns)
        return pd.DataFrame(knn(n_neighbors=k, weights='uniform').fit_transform(df), columns=colNames)

    df = dropSingleValues(df)
    df = dropMissingThirty(df)
    df = numerizor(df)
    df = knnMissingValues(df)
    df = minMax(df)

    return df

import os

fileList = []
fileCounter = 1
for root, dirs, files in os.walk(r'Datasets'):
    for file in files:
        transformedData = preprocessing(pd.read_csv(r'Datasets\\'+file))
        print('File ' + str(fileCounter) + ' transformed.')
        fileList.append([file, 'dataset '+str(fileCounter)])
        transformedData.to_csv(r'TransformedDatasets\dataset '+str(fileCounter), sep=',')
        print('File ' + str(fileCounter) + ' saved normally.')
        fileCounter += 1


fileList = np.asarray(fileList)
fileList = pd.DataFrame(fileList, columns=['Original Datasets', 'Transformed Datasets'])
        
fileList.to_csv('MetaDataNames', sep=',')







        

