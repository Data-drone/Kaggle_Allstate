# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 17:23:26 2016

@author: Brian
"""

"""
stacker model with feats

"""
import pickle
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle

feats_dir = '../Dropbox/Kaggle_AllState/'
ensemble_dir = '../Dropbox/Kaggle_AllState/ensemble'
shift = 200
NFOLDS = 10
SEED = 10

# read in the feat set
train_feats = pickle.load(open(feats_dir + 'train_model_5.pkl', 'rb') )
test_feats = pickle.load(open(feats_dir + 'test_model_5.pkl', 'rb') )

# make the models
train_x = train_feats.drop(['loss', 'id'], axis = 1)
train_y = np.log(train_feats['loss'] + shift)
test_x = test_feats.drop(['loss', 'id'], axis = 1)

n_train = train_x.shape[0]
n_test = test_x.shape[0]
# kfold

x_train= np.array(train_x)
x_test= np.array(test_x)

kf = KFold(n_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, np.log(y_train))

    def predict(self, x):
        return np.exp(self.clf.predict(x))

        
def get_oof(clf):
    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_skf = np.empty((NFOLDS, n_test))

    for i, (train_index, test_index) in enumerate(kf):
        print(str(i))
        x_tr = x_train[train_index]
        y_tr = train_y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

et_params = {
    'n_jobs': -1,
    'n_estimators': 400,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': -1,
    'n_estimators': 400,
    'max_features': 0.2,
    'max_depth': 8,
    'min_samples_leaf': 2,
}


#xg = XgbWrapper(seed=SEED, params=xgb_params)

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)

#xg_oof_train, xg_oof_test = get_oof(xg)
print('ExtraTreesRegressor')
et_oof_train, et_oof_test = get_oof(et)
print('RandomForestRegressor')
rf_oof_train, rf_oof_test = get_oof(rf)

print("ET-CV: {}".format(mean_absolute_error(train_y, et_oof_train)))
print("RF-CV: {}".format(mean_absolute_error(train_y, rf_oof_train)))

pickle.dump(et_oof_train, open(ensemble_dir + "et_oof_train.pkl", "wb"))
pickle.dump(et_oof_test, open(ensemble_dir + "et_oof_test.pkl", "wb"))
pickle.dump(rf_oof_train, open(ensemble_dir + "rf_oof_train.pkl", "wb"))
pickle.dump(rf_oof_test, open(ensemble_dir + "rf_oof_test.pkl", "wb"))

"""
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=np.log(y_train))
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
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True, feval=xg_eval_mae, maximize=False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))
submission.to_csv('xgstacker_starter_v2.sub.csv', index=None)
"""