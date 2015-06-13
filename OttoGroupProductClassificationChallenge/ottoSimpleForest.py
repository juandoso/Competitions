'''
Created on 11/03/2015

@author: Juandoso
'''
import pandas as pd
import os
from sklearn import preprocessing, feature_extraction

data_dir = 'F:/OttoGroupProductClassificationChallenge/'
#data_dir = 'data/'

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Grid Search for optimal classifier parameters
from sklearn.grid_search import GridSearchCV
from OttoGroupProductClassificationChallenge.benchmark import logloss_mc
from sklearn.metrics import make_scorer
multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
#scoring=multiclass_log_loss
param_grid = {'n_estimators': range(900, 1001, 25)}
forest = GridSearchCV(RandomForestClassifier(n_jobs=2), param_grid)
forest = forest.fit(train,labels)
print("Best estimator found by grid search:")
print(forest.best_estimator_)

# forest = RandomForestClassifier(n_estimators=900, n_jobs=2)
# forest = forest.fit(train,labels)

print 'Predicting...'
results = forest.predict_proba(test)

print "Writing output file"
preds = pd.DataFrame(results, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv(os.path.join(data_dir,"results.csv"), index_label='id')
print 'Done.'

print("... in %0.3fs" % (time() - t0))