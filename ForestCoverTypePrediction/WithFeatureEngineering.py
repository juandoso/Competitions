'''
Created on 11/03/2015

@author: Juandoso
'''

import csv
import pandas as pd
import os
from math import sqrt

data_dir = 'F:/ForestCoverTypePrediction/'
#data_dir = 'data/'

from time import time
t0 = time()

print "Importing data..."
data_df = pd.read_csv(os.path.join(data_dir,'train.csv'), header=0)
Y = data_df['Cover_Type'].as_matrix()
X_id = data_df['Id'].as_matrix()
data_df.drop(['Cover_Type', 'Id'], axis=1, inplace=True)
contest_df = pd.read_csv(os.path.join(data_dir,'test.csv'), header=0)
contest_id = contest_df['Id'].as_matrix()
contest_df.drop(['Id'], axis=1, inplace=True) 

print "Some Feature Engineering..."

#As we know about that Aspect is in degrees azimuth, we can try to shift it at 180
def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

data_df['Aspect2'] = data_df.Aspect.map(r)
contest_df['Aspect2'] = contest_df.Aspect.map(r)

#Above or below water level
data_df['Highwater'] = data_df.Vertical_Distance_To_Hydrology < 0
contest_df['Highwater'] = contest_df.Vertical_Distance_To_Hydrology < 0
    
#Relations between variables
#Elevation - Distance to water
data_df['EVDtH'] = data_df.Elevation-data_df.Vertical_Distance_To_Hydrology
contest_df['EVDtH'] = contest_df.Elevation-contest_df.Vertical_Distance_To_Hydrology
data_df['EHDtH'] = data_df.Elevation-data_df.Horizontal_Distance_To_Hydrology*0.2
contest_df['EHDtH'] = contest_df.Elevation-contest_df.Horizontal_Distance_To_Hydrology*0.2
    
#Other nonlinear relations
data_df['Distance_to_Hydrolody'] = (data_df['Horizontal_Distance_To_Hydrology']**2+data_df['Vertical_Distance_To_Hydrology']**2)**0.5
contest_df['Distance_to_Hydrolody'] = (contest_df['Horizontal_Distance_To_Hydrology']**2+contest_df['Vertical_Distance_To_Hydrology']**2)**0.5

data_df['Hydro_Fire_1'] = data_df['Horizontal_Distance_To_Hydrology']+data_df['Horizontal_Distance_To_Fire_Points']
contest_df['Hydro_Fire_1'] = contest_df['Horizontal_Distance_To_Hydrology']+contest_df['Horizontal_Distance_To_Fire_Points']

data_df['Hydro_Fire_2'] = abs(data_df['Horizontal_Distance_To_Hydrology']-data_df['Horizontal_Distance_To_Fire_Points'])
contest_df['Hydro_Fire_2'] = abs(contest_df['Horizontal_Distance_To_Hydrology']-contest_df['Horizontal_Distance_To_Fire_Points'])

data_df['Hydro_Road_1'] = abs(data_df['Horizontal_Distance_To_Hydrology']+data_df['Horizontal_Distance_To_Roadways'])
contest_df['Hydro_Road_1'] = abs(contest_df['Horizontal_Distance_To_Hydrology']+contest_df['Horizontal_Distance_To_Roadways'])

data_df['Hydro_Road_2'] = abs(data_df['Horizontal_Distance_To_Hydrology']-data_df['Horizontal_Distance_To_Roadways'])
contest_df['Hydro_Road_2'] = abs(contest_df['Horizontal_Distance_To_Hydrology']-contest_df['Horizontal_Distance_To_Roadways'])

data_df['Fire_Road_1'] = abs(data_df['Horizontal_Distance_To_Fire_Points']+data_df['Horizontal_Distance_To_Roadways'])
contest_df['Fire_Road_1'] = abs(contest_df['Horizontal_Distance_To_Fire_Points']+contest_df['Horizontal_Distance_To_Roadways'])

data_df['Fire_Road_2'] = abs(data_df['Horizontal_Distance_To_Fire_Points']-data_df['Horizontal_Distance_To_Roadways'])
contest_df['Fire_Road_2'] = abs(contest_df['Horizontal_Distance_To_Fire_Points']-contest_df['Horizontal_Distance_To_Roadways'])

###
X = data_df.as_matrix()
contestX = contest_df.as_matrix()
n_features = int(sqrt(X.shape[1]))

print "Training..."
from sklearn.grid_search import GridSearchCV

#Random Forest param search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
param_grid = {'n_estimators': range(870, 880, 1)}
forest = GridSearchCV(ExtraTreesClassifier(), param_grid)
#, n_jobs=-1

# from sklearn.ensemble import GradientBoostingClassifier
# param_grid = {'n_estimators': range(400, 1201, 200),
#               'learning_rate': [0.05, 0.1],
#               'subsample': [0.5, 1.0],
#               'max_features': [n_features/2, n_features],
#               'max_depth': [2, 3, 4]}
# forest = GridSearchCV(GradientBoostingClassifier(), param_grid, n_jobs=7)

forest = ExtraTreesClassifier(n_estimators=876)
forest = forest.fit(X,Y)

# print("Best estimator found by grid search:")
# print(forest.best_estimator_)
# print(forest.best_params_)
# print(forest.best_score_)
# print("Scores: ")
# print(forest.grid_scores_)

print 'Predicting...'
results = forest.predict(contestX).astype(int)
print results.shape

print "Writing output file"
predictions_file = open(os.path.join(data_dir,"results2.csv"), "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Cover_Type"])
open_file_object.writerows(zip(contest_id, results))
predictions_file.close()
print 'Done.'

print("... in %0.3fs" % (time() - t0))