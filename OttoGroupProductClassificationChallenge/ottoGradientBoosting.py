'''
Created on 24/03/2015

@author: Juandoso
'''

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from operator import itemgetter
from scipy.stats import randint as sp_randint, uniform


data_dir = 'F:/OttoGroupProductClassificationChallenge/'
#data_dir = 'data/'

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

from time import time
t0 = time()

print "Importing data..."
train = pd.read_csv(os.path.join(data_dir,'train.csv'), header=0)
test = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0)
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))

labels = train['target'].as_matrix()
train.drop(['target', 'id'], axis=1, inplace=True)
test_id = test['id'].as_matrix()
test.drop(['id'], axis=1, inplace=True)

lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


print "Training..."

from sklearn.ensemble import GradientBoostingClassifier

#Grid Search for optimal classifier parameters
from sklearn.grid_search import GridSearchCV
 
 #scoring=multiclass_log_loss
param_grid = {'n_estimators': range(100, 2001, 100)}
 # specify parameters and distributions to sample from
#  param_dist = {"max_depth": [3, None],
#                "max_features": sp_randint(1, 11),
#                "min_samples_split": sp_randint(1, 11),
#                "min_samples_leaf": sp_randint(1, 11),
#                "subsample": [0.8, 1.0],
#                "learning_rate": uniform()}

forest = GridSearchCV(GradientBoostingClassifier(), param_grid)
forest = forest.fit(train,labels)
print("Best estimator found by grid search:")
print(forest.best_estimator_)
print(forest.best_score_)

# forest = GradientBoostingClassifier()
# forest = forest.fit(train,labels)

print 'Predicting...'
results = forest.predict_proba(test)

print "Writing output file"
preds = pd.DataFrame(results, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv(os.path.join(data_dir,"results.csv"), index_label='id')
print 'Done.'

print("... in %0.3fs" % (time() - t0))