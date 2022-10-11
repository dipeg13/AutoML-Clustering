import os
import numpy as np
import pandas as pd
from random_search import random_search_internal, random_search_external

os.environ['R_HOME'] = r'C:\Program Files\R\R-4.2.1'

motherRoot = r'C:\Users\user\Desktop\Δημήτρης\ExhaustiveSearch-Clustering-main\Datasets'

counter = 0
"""
for filename in os.listdir(motherRoot):
    print(filename)
    folder_path = motherRoot + '\\' + filename
    for child in os.listdir(folder_path):
        print(child)
        if os.path.isfile(folder_path + '\\' + child + '\\raw_data\\target.csv'):
            df = pd.read_csv(folder_path + '\\' + child + '\\clean_data.csv', header=None)
            target = pd.read_csv(folder_path + '\\' + child +'\\raw_data\\target.csv', header=None).to_numpy().flatten()

            results = random_search_external(df, target, 300)
            results.to_csv(filename+'-'+child+'-randomSearch.csv', index=False)
        else:
            df = pd.read_csv(folder_path + '\\' + child + '\\clean_data.csv', header=None)

            results = random_search_internal(df, 300)
            results.to_csv(filename+'-'+child+'-randomSearch.csv', index=False)
"""

for filename in os.listdir(motherRoot):
    print(filename)
    if filename == 'uci':
        folder_path = motherRoot + '\\' + filename
        for child in os.listdir(folder_path):
            print(child)
            if os.path.isfile(folder_path + '\\' + child + '\\raw_data\\target.csv'):
                df = pd.read_csv(folder_path + '\\' + child + '\\clean_data.csv', header=None)
                target = pd.read_csv(folder_path + '\\' + child +'\\raw_data\\target.csv', header=None).to_numpy().flatten()

                results = random_search_external(df, target, 300)
                results.to_csv(filename+'-'+child+'-randomSearch.csv', index=False)
            else:
                df = pd.read_csv(folder_path + '\\' + child + '\\clean_data.csv', header=None)

                results = random_search_internal(df, 300)
                results.to_csv(filename+'-'+child+'-randomSearch.csv', index=False)
