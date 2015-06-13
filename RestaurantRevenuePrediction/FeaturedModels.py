'''
Created on 29/03/2015

@author: Juandoso
'''

from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer

from time import time
t0 = time()

data_dir = 'F:/RestaurantRevenuePrediction/data/'
#data_dir = 'data/' 

print "Importing data..."
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')

Y = np.log10(train['revenue'].as_matrix())
train.drop(["revenue"], axis=1, inplace=True)

#Transform Open Dates to Age
def diff_dates_2015(date_x):
    date_format = "%m/%d/%Y"
    x = datetime.strptime(date_x, date_format)
    y = datetime.strptime('01/01/2015', date_format)
    delta = y - x
    return delta.days/30

def rmse(yp, y):
    return np.sqrt(np.sum(np.square(y - yp)) / len(y))

train["months_old"] = train["Open Date"].apply(lambda x: diff_dates_2015(x))
train["years_since_1990"] = train["months_old"] / 12 
test["months_old"] = test["Open Date"].apply(lambda x: diff_dates_2015(x))
test["years_since_1990"] = test["months_old"] / 12
train.drop(["Open Date"], axis=1, inplace=True)
test.drop(["Open Date"], axis=1, inplace=True)

#Cleaning Data
train.drop(["City"], axis=1, inplace=True)
test.drop(["City"], axis=1, inplace=True)

mb_index = test.Type == "MB" 
type_mode = test.Type.dropna().mode().values
test.Type[ mb_index ] = type_mode

#Transform Categorical features to One Hot
vec = DictVectorizer()
featuresToVect = ["City Group","Type"]
X_train_cat=vec.fit_transform(train[featuresToVect].T.to_dict().values()).todense()
X_test_cat = vec.transform(test[featuresToVect].T.to_dict().values()).todense()

train.drop(featuresToVect, axis=1, inplace=True)
test.drop(featuresToVect, axis=1, inplace=True)

min_max_scaler = MinMaxScaler()

#X_train_num = train.as_matrix()
#X_test_num = test.as_matrix()
X_train_num = min_max_scaler.fit_transform(train.as_matrix())
X_test_num = min_max_scaler.fit_transform(test.as_matrix())

X_train = np.hstack((X_train_cat, X_train_num))
X_test = np.hstack((X_test_cat, X_test_num))
n_features = X_train.shape[1]

print "Training..."

#Forest Model
# from sklearn.cross_validation import LeaveOneOut
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import mean_squared_error, make_scorer
# mse = make_scorer(mean_squared_error, greater_is_better=False)
# param_grid = {'n_estimators': range(1500, 2101, 50)}
# model = GridSearchCV(AdaBoostRegressor(loss="square"), param_grid, cv=LeaveOneOut(137), scoring=mse)
# model.fit(X_train,Y)
# print("Best estimator found by grid search:")
# print(model.best_estimator_)
# print(model.best_params_)
# print(model.best_score_)

# model = AdaBoostRegressor(n_estimators=1700)
# model.fit(X_train,Y)

#Linear Model
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import LeaveOneOut
#model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LassoLarsCV(max_iter=5000, n_jobs=-1, cv=LeaveOneOut(137)))])
model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=100000, cv=LeaveOneOut(137))
model.fit(X_train,Y)
print model.alpha_
print model.l1_ratio_

#Neighbors
# model = KNeighborsClassifier(n_neighbors = 1)
# model.fit(X_train,Y)

print 'Predicting...'
prediction = model.predict(X_test)

print "Writing output file..."
submission = pd.read_csv(data_dir + 'sampleSubmission.csv')
submission['Prediction'] = np.power(prediction, 10)
submission.to_csv(os.path.join(data_dir,"results.csv"),index=False)

print 'Done.'
print("... in %0.3fs" % (time() - t0))
