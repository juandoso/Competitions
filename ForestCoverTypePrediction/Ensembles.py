'''
Created on 5/05/2015

@author: Juandoso
'''

import csv
import pandas as pd
import os
from scipy.stats import mode
import numpy as np
from pandas.core.frame import DataFrame

data_dir = 'F:/ForestCoverTypePrediction/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))
results = pd.DataFrame(index=sample.Id.values, columns=sample.columns[1:])
responses = pd.DataFrame()
for i in [1,2,3]:
    df = pd.read_csv(os.path.join(data_dir,'ensemble2/%s.csv' % (i)), header=0)
    responses[i] = df["Cover_Type"]

print "Finding mode..."
res = responses.as_matrix()
resmax = mode(res, axis=1)
results.Cover_Type = resmax[0]

preds = pd.DataFrame(results["Cover_Type"], index=sample.Id.values, columns=sample.columns[1:])
preds.to_csv(os.path.join(data_dir,"ensemble_results.csv"), index_label='Id')