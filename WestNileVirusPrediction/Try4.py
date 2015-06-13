'''
Created on 28/04/2015

@author: Juandoso
'''
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler


data_dir = 'F:/WestNileVirusPrediction/data/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
train = pd.read_csv(os.path.join(data_dir,'train3.csv'), header=0, parse_dates=[0])
test = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0, parse_dates=[1])
weather = pd.read_csv(os.path.join(data_dir,'weather_info_averages4.csv'), header=0, parse_dates=[-1])
spray = pd.read_csv(os.path.join(data_dir,'spray.csv'), header=0)
sample = pd.read_csv(os.path.join(data_dir,'sampleSubmission.csv'))

test_id = test["Id"]
test.drop("Id", axis=1, inplace=True)

#Merge with weather data
train = pd.merge(train, weather, left_on='Date', right_on='Date', how="inner")
test = pd.merge(test, weather, left_on='Date', right_on='Date', how="inner")
#train.to_csv(os.path.join(data_dir,"train_merged.csv"))
#test.to_csv(os.path.join(data_dir,"test_merged.csv"))

#Cleaning address data
address_columns = ["Address", "AddressNumberAndStreet", "AddressAccuracy"]
train.drop(address_columns, axis=1, inplace=True)
test.drop(address_columns, axis=1, inplace=True)

coordinates_columns = ["Latitude", "Longitude"]
geo_lat = (train["Latitude"].min(), train["Latitude"].max()) 
geo_lon = (train["Longitude"].min(), train["Longitude"].max())

'''
max_geo_lat: 42.01743, min_geo_lat: 41.644612, diff: 0.372818
min_geo_lon: -87.930995, max_geo_lon: -87.531635, diff: 0.39936
'''
def geo_to_sector(geo_point, geo_tuple, num_sectors=30):
    gmin = geo_tuple[0]
    gmax = geo_tuple[1]
    diff = gmax-gmin
    delta = diff/num_sectors
    sector = (geo_point - gmin) / delta
    return int(sector)
 
coordinates_sector_columns = ["latitude_sector", "longitude_sector"]
train["latitude_sector"] = train["Latitude"].apply(lambda x: geo_to_sector(geo_point=x, geo_tuple=geo_lat))
train["longitude_sector"] = train["Longitude"].apply(lambda x: geo_to_sector(geo_point=x, geo_tuple=geo_lon))
test["latitude_sector"] = test["Latitude"].apply(lambda x: geo_to_sector(geo_point=x, geo_tuple=geo_lat))
test["longitude_sector"] = test["Longitude"].apply(lambda x: geo_to_sector(geo_point=x, geo_tuple=geo_lon))
train.drop(coordinates_columns, axis=1, inplace=True)
test.drop(coordinates_columns, axis=1, inplace=True)

address_columns_split = ["Block", "Street"]
train.drop(address_columns_split, axis=1, inplace=True)
test.drop(address_columns_split, axis=1, inplace=True)


geo_features = coordinates_sector_columns
vec2 = DictVectorizer()
X_dir = pd.concat([train[geo_features], test[geo_features]])
vec2.fit(X_dir.T.to_dict().values())


train["year"] = train.Date.apply(lambda x: x.year)
train["week"] = train.Date.apply(lambda x: x.isocalendar()[1])
test["year"] = test.Date.apply(lambda x: x.year - 1)
test["week"] = test.Date.apply(lambda x: x.isocalendar()[1])
train.drop(['Date'], axis = 1, inplace=True)
test.drop(['Date'], axis = 1, inplace=True)

min_max_scaler = MinMaxScaler()

#Data is now ready 
# print pd.isnull(train)
# print pd.isnull(test)
#train.to_csv(os.path.join(data_dir,"train_ready.csv"))
#test.to_csv(os.path.join(data_dir,"test_ready.csv"))

    
featuresCateg = ["Species", "Trap", "week", "year"]
vec = DictVectorizer()
X_cat = pd.concat([train[featuresCateg], test[featuresCateg]])
vec.fit(X_cat.T.to_dict().values())


def regression_NumMosquitos(Xtr, ytr, Xte):
    from sklearn.linear_model import Ridge, RidgeCV
    #model_nm = RidgeCV(alphas=range(200, 401, 10), cv=5)
    model_nm = Ridge(alpha = 340)
    model_nm = model_nm.fit(Xtr, ytr)
    results_nm = model_nm.predict(Xte)
    return results_nm

def obtain_best_model(X, y):
    from sklearn.grid_search import GridSearchCV
    
    from sklearn.ensemble import ExtraTreesClassifier
    param_grid = {'n_estimators': range(2,41,2),
                  'criterion': ['entropy'],
                  'min_samples_split': [1,2],
                  'min_samples_leaf': [5,6,7,8]}
    forest = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=10, n_jobs=1, scoring='roc_auc')
    
    forest = forest.fit(X,y)
    print("Best parameters found by grid search: ")
    print(forest.best_params_)
    print("Best score found by grid search: ")
    print(forest.best_score_)
    return forest, forest.best_params_, forest.best_estimator_

#     forest = RandomForestClassifier(min_samples_split= 2, n_estimators= 35, criterion= 'entropy', min_samples_leaf= 5)
#     forest.fit(X, y)
#     return forest, None, forest

def predict_with_best_model(estimator, xtrain, ytrain, xtest):
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier(base_estimator=estimator, n_estimators=10, max_samples=0.9, max_features=0.9, n_jobs=1, 
                              bootstrap=False, bootstrap_features=False, oob_score=False)
    model = model.fit(xtrain,ytrain)
    y = model.predict_proba(xtest)
#     print("Bagging score with oob estimates: ")
#     print model.oob_score_
    print ("Model used: ")
    print model.base_estimator_
    return y 

tr = train
te = test
# tr.drop("year", axis=1, inplace=True)
# te.drop("year", axis=1, inplace=True)
    
print "Engineering Features..."
    
y_nm = tr["NumMosquitos"]
y_wn = tr["WnvPresent"]
y_wn[ y_wn>0 ] = 1 
tr.drop(["NumMosquitos", "WnvPresent"], axis=1, inplace=True)
    
tr_cat = vec.transform(tr[featuresCateg].T.to_dict().values()).todense()
te_cat = vec.transform(te[featuresCateg].T.to_dict().values()).todense()
tr.drop(featuresCateg, axis=1, inplace=True)
te.drop(featuresCateg, axis=1, inplace=True)
    
tr_geo = vec2.transform(tr[geo_features].T.to_dict().values()).todense()
te_geo = vec2.transform(te[geo_features].T.to_dict().values()).todense()
tr.drop(geo_features, axis=1, inplace=True)
te.drop(geo_features, axis=1, inplace=True)

tr_num = min_max_scaler.fit_transform(tr.as_matrix())
te_num = min_max_scaler.fit_transform(te.as_matrix())
    
Xtrain = np.hstack((tr_num, tr_cat, tr_geo))
Xtest = np.hstack((te_num, te_cat, te_geo))
        
print "Training ..."
    
results_nm = regression_NumMosquitos(Xtrain, y_nm, Xtest)
Xtrain.NumMosquitos = y_nm
Xtest.NumMosquitos = results_nm
     
model_wn, params, estimator = obtain_best_model(Xtrain, y_wn)
     
print "Predicting ..."
    
#results_wn = model_wn.predict_proba(Xtest)
results_wn = predict_with_best_model(estimator=estimator, xtrain=Xtrain, ytrain=y_wn, xtest=Xtest)

results = DataFrame(results_wn)
results1 = 1.0 - results[0]
#results1 = results[1]

print "Writing output file"
preds = pd.DataFrame(results1.as_matrix(), index=sample.Id.values, columns=sample.columns[1:])
preds.to_csv(os.path.join(data_dir,"results.csv"), index_label='Id')

print 'Done.'

print("... in %0.3fs" % (time() - t0))
