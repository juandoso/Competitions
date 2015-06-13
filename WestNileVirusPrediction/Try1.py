'''
Created on 22/04/2015

@author: Juandoso
'''

import pandas as pd
import os
from pandas.core.frame import DataFrame

data_dir = 'F:/WestNileVirusPrediction/data/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
train = pd.read_csv(os.path.join(data_dir,'train.csv'), header=0, parse_dates=[0])
test = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0, parse_dates=[1])
weather = pd.read_csv(os.path.join(data_dir,'weather.csv'), header=0)
spray = pd.read_csv(os.path.join(data_dir,'spray.csv'), header=0)
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))

#Processing basic train data
train.drop(["Address", "AddressNumberAndStreet"], axis=1, inplace=True)
test.drop(["Address", "AddressNumberAndStreet"], axis=1, inplace=True)
test_id = test["Id"]
test.drop("Id", axis=1, inplace=True)

train["year"] = train.Date.apply(lambda x: x.year)
train["month"] = train.Date.apply(lambda x: x.month)
train["day"] = train.Date.apply(lambda x: x.day)
test["year"] = test.Date.apply(lambda x: x.year - 2000)
test["month"] = test.Date.apply(lambda x: x.month)
test["day"] = test.Date.apply(lambda x: x.day)
train.drop("Date", axis=1, inplace=True)
test.drop("Date", axis=1, inplace=True)

y_nm = train["NumMosquitos"]
y_wn = train["WnvPresent"]
train.drop(["NumMosquitos", "WnvPresent"], axis=1, inplace=True)

train.drop(["Street", "Trap", "Species"], axis=1, inplace=True)
test.drop(["Street", "Trap", "Species"], axis=1, inplace=True)

# featuresCateg = ["Species", "Street", "Trap"]
# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer()
# train_cat = vec.fit_transform(train[featuresCateg].T.to_dict().values()).todense()
# test_cat = vec.fit_transform(test[featuresCateg].T.to_dict().values()).todense()
# train.drop(featuresCateg, axis=1, inplace=True)
# test.drop(featuresCateg, axis=1, inplace=True)

train_num = train.as_matrix()
test_num = test.as_matrix()
# import numpy as np
# Xtrain = np.hstack((train_num, train_cat))
# Xtest = np.hstack((test_num, test_cat))
Xtrain = train_num
Xtest = test_num

print "Training..."
from sklearn.linear_model import ElasticNetCV
model_nm = ElasticNetCV()
model_nm.fit(Xtrain, y_nm)

from sklearn.ensemble import RandomForestClassifier
model_wn = RandomForestClassifier()
model_wn.fit(Xtrain, y_wn)

print 'Predicting...'

#results_nm = model_nm.predict(Xtest)
#results_nm = int(results_nm)
results_wn = model_wn.predict_proba(Xtest)
results = DataFrame(results_wn)
results1 = results[1].as_matrix()

print "Writing output file"
preds = pd.DataFrame(results1, index=sample.Id.values, columns=sample.columns[1:])
preds.to_csv(os.path.join(data_dir,"results.csv"), index_label='Id')

print "Estimating scores..."
from sklearn import cross_validation
import numpy
scores = cross_validation.cross_val_score(model_wn, Xtrain, y_wn, cv=5, scoring='roc_auc')
score = numpy.average(scores)
print "CV ROC AUC average Score: %s" % (score)

print 'Done.'

print("... in %0.3fs" % (time() - t0))