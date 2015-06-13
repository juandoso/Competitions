'''
Created on 11/03/2015

@author: Juandoso
'''
import csv
import pandas as pd
import os

data_dir = 'F:/ForestCoverTypePrediction/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
data_df = pd.read_csv(os.path.join(data_dir,'train.csv'), header=0)
Y = data_df['Cover_Type'].as_matrix()
X_id = data_df['Id'].as_matrix()
data_df.drop(['Cover_Type', 'Id'], axis=1, inplace=True)
X = data_df.as_matrix()
contest_df = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0)
contest_id = contest_df['Id'].as_matrix()
contest_df.drop(['Id'], axis=1, inplace=True) 
contestX = contest_df.as_matrix()

print "Training..."
from sklearn.grid_search import GridSearchCV

#Random Forest param search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier
param_grid = {'n_estimators': range(100, 1001, 100)}
forest = GridSearchCV(ExtraTreesClassifier(), param_grid)

forest = forest.fit(X,Y)
print("Best estimator found by grid search:")
print(forest.best_estimator_)

print 'Predicting...'
results = forest.predict(contestX).astype(int)
print results.shape

print "Writing output file"
predictions_file = open(os.path.join(data_dir,"results.csv"), "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Cover_Type"])
open_file_object.writerows(zip(contest_id, results))
predictions_file.close()
print 'Done.'

print("... in %0.3fs" % (time() - t0))