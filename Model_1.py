# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 23:23:27 2016

@author: Brian
"""

"""
Build model 1 script

"""
import os
import pandas as pd
import numpy as np
"""
feat table_1
"""

#### standard functions
def Find_Low_Var(frame):
    sizes = []
    low_var_col = []
    for column in frame:
        data_sets = frame[column]
        uniques = data_sets.unique() # note the brackets
        # print the pivot to show distributions
        # will error
        if (column != 'id'):
            pd_Table = pd.pivot_table(train, values = 'id', index=column, aggfunc='count').apply(lambda x: np.round(np.float(x)/len(train)*100, 2))
            #print (pd_Table )
            if max(pd_Table) > 99:
                low_var_col.append(column)
        #variance = np.var(data_sets)
        #print(uniques.size)
        #print(column)
        
        sizes.append(uniques.size)
    #print(len(uniques))
    return(sizes, low_var_col)



data_path = 'Data'
os.listdir('./Data/')
train = pd.read_csv('./Data/train.csv')

### 2 tyoes if columns the continuous and the categorical
# lets separate these

cate_cols = [col for col in train.columns if 'cat' in col]
continuous_cols = [col for col in train.columns if 'cont' in col]

categorical = train[cate_cols]
sizes, low_var_cols_to_drop = Find_Low_Var(categorical)
categorical_to_keep = categorical.drop(low_var_cols_to_drop, axis = 1)

### check how big the one hot is first
OneHot = pd.get_dummies(categorical_to_keep)


## try feature hasher again
from sklearn.feature_extraction import FeatureHasher
FH = FeatureHasher(n_features = 1000, input_type = 'dict')
hashed_Feat = FH.transform(categorical_to_keep.to_dict(orient='records'))
#dense_Feat = hashed_Feat.todense()

### make into categories

continuous = train[continuous_cols]

## id and target columns
id_target = train[['id', 'loss']]


## quick test 1
frame = [continuous, OneHot]
merge = pd.concat(frame, axis = 1)

"""

train test splitting

"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( merge, np.log(id_target.loss), test_size=0.4, random_state=0)
assert X_train.shape[0] + X_test.shape[0] == continuous.shape[0]


# model 1 # like 
# 8917348 RMSE for log 
# 8116237 for normal 
#from sklearn import linear_model
#reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
#reg.fit(X_train, y_train)
#result = reg.predict(X_test)
 
# model 2
# 9069687 rmse for 100 regressors
# with one hots vars 4332759 rmse
#from sklearn.ensemble import RandomForestRegressor
#clf = RandomForestRegressor(n_estimators = 400, criterion='mse', verbose = 1, n_jobs = 7)
#clf.fit(X_train, y_train)
#result = clf.predict(X_test)

# model 3 # default xgb was 4389784
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgb_mod = XGBRegressor(max_depth = 10, learning_rate = 0.25, n_estimators = 150)
xgb_mod.fit(X_train, y_train)
result = xgb_mod.predict(X_test)

# score
from sklearn.metrics import mean_squared_error
mean_squared_error(np.exp(y_test), np.exp(result) )



import matplotlib.pyplot as plt
y_test.hist()
plt.hist(result, bins='auto')