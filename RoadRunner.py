import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from autoclust.baseline_optimization import ParameterTuner

os.environ['R_HOME'] = r'C:\Program Files\R\R-4.2.1'

motherRoot = r'C:\Users\Administrator\Desktop\ExhaustiveSearch-Clustering-main\Datasets'

for filename in os.listdir(motherRoot):
    print(filename)
    for child in os.listdir(motherRoot+'\\'+filename):
        if filename=='openml':
            if os.path.isfile(motherRoot+'\\'+filename+'\\'+child+'\\'+'\\raw_data\\target.csv'):
                df = pd.read_csv(motherRoot+'\\'+filename+'\\'+child+'\\clean_data.csv', header=None)

                target = pd.read_csv(motherRoot+'\\'+filename+'\\'+child+'\\raw_data\\target.csv', header=None).to_numpy().flatten()
                
                pm = ParameterTuner(data=df, search_space="autoclust/SearchSpace/Model_Specifications_GridSearch.json", external=(True, target))

                temp_df = pm.grid_search(cvi="r", logging=False)

                masterdf = pd.DataFrame()
                masterdf = masterdf.append(temp_df)

                #masterdf['Algorithm'].unique()
                masterdf.to_csv(filename+'-'+child+'-gridSearch.csv', index=False)

            else:
                df = pd.read_csv(motherRoot+'\\'+filename+'\\'+child+'\\clean_data.csv', header=None)

                pm = ParameterTuner(data=df, search_space="autoclust/SearchSpace/Model_Specifications_GridSearch.json")

                temp_df = pm.grid_search(cvi="r", logging=False)

                masterdf = pd.DataFrame()
                masterdf = masterdf.append(temp_df)

                #masterdf['Algorithm'].unique()
                masterdf.to_csv(filename+'-'+child+'-gridSearch.csv', index=False)
