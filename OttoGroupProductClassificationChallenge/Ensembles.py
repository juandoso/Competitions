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

data_dir = 'F:/OttoGroupProductClassificationChallenge/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))
sample["Class_1"] = sample["Class_1"]  - 1
clases = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
models = [1,2,3,4,5,6,7,8]
#,5,6,7,8,9,10]
weights = [0.47097, 0.47157, 0.48362, 0.48961, 0.48983, 0.49184 , 0.49575, 0.49772]
#weights = [0.47157, 0.48362, 0.49184, 0.49575]
#, 0.55229, 0.55338, 0.55338, 0.56321, 0.56962, 0.59352]
one = np.ones(len(weights))
weights = one - weights
#weights = (weights*10)**2
for i in models:
    df = pd.read_csv(os.path.join(data_dir,'ensemble2/%s.csv' % (i)), header=0)
    #sample[clases] += df[clases]
    w = weights[i-1]
    sample[clases] += df[clases] * w

print "Finding average..."
#sample[clases] = sample[clases]/len(models)
sample[clases] = sample[clases]/sum(weights)
sample.to_csv(os.path.join(data_dir,"ensemble_results.csv"), index_label='id', index=False)
