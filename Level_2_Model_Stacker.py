# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 00:31:50 2016

@author: Brian
"""

### quick explore
import pickle
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import xgboost as xgb

# local
directory = '../Dropbox/Kaggle_AllState/'
ensemble_dir = '../Dropbox/Kaggle_AllState/ensemble'
feats_dir = '../Dropbox/Kaggle_AllState/'
DATA_DIR = directory
shift = 200
    
import platform
if (platform.system() == 'Windows'):
    directory = 'E:/Dropbox/Dropbox/Kaggle_AllState/'
    ensemble_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/ensemble'
    feats_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/'
    DATA_DIR = directory

    
SEED = 0

y_train = np.array(pd.read_csv(directory + 'train.csv').loss.ravel(), dtype=np.float64)
SUBMISSION_FILE =     "{0}/sample_submission.csv".format(DATA_DIR)
    #### Load
et_oof_test = pickle.load(open(directory + 'ensemblerf_oof_test.pkl', 'rb') )
et_oof_train = pickle.load(open(directory + 'ensemblerf_oof_train.pkl', 'rb') )

xgb_oof_test = pickle.load(open(directory + 'ensemblexgb_oof_test_2.pkl', 'rb') )
xgb_oof_test = np.log(xgb_oof_test + shift)
xgb_oof_train = pickle.load(open(directory + 'ensemblexgb_oof_train_2.pkl', 'rb') )
# might just be fix for this file
xgb_oof_train = xgb_oof_train.reshape(-1,1)

rf_oof_test = pickle.load(open(directory + 'ensemblerf_oof_test.pkl', 'rb') )
rf_oof_train = pickle.load(open(directory + 'ensemblerf_oof_train.pkl', 'rb') )

# xgb set is off
print("XG-CV: {}".format(mean_absolute_error(y_train, np.exp(xgb_oof_train) - shift ) ) )

print("ET-CV: {}".format(mean_absolute_error(y_train, np.exp(et_oof_train))))
#print("ET-CV: {}".format(mean_absolute_error(np.log(y_train), et_oof_train)))
print("RF-CV: {}".format(mean_absolute_error(y_train, np.exp(rf_oof_train))))

"""
import pandas as pd
test = pd.read_csv(directory + 'submission_5fold-average-xgb_fairobj_1130.839299_2016-12-06-13-37.csv')
"""



x_train = np.concatenate((xgb_oof_train, et_oof_train, rf_oof_train), axis=1)
x_test = np.concatenate((xgb_oof_test, et_oof_test, rf_oof_test), axis=1)


print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=np.log(y_train + shift))
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y - shift), np.exp(yhat - shift))

res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True, feval=xg_eval_mae, maximize=False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest)) - shift
submission.to_csv("{0}/xgstacker_starter_v2_tweak.csv".format(DATA_DIR), index=None)

