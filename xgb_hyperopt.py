# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 23:14:08 2016

@author: Brian
"""

####  Hyperopt_test
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from sklearn.cross_validation import KFold

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

if __name__ == "__main__":

    shift = 200
    
    print('\nStarted')
    #directory = '../input/'
    directory = '../Dropbox/Kaggle_AllState/'
    ensemble_dir = '../Dropbox/Kaggle_AllState/ensemble/'
    feats_dir = '../Dropbox/Kaggle_AllState/'
 
    # local
    import platform
    if (platform.system() == 'Windows'):
        directory = 'E:/Dropbox/Dropbox/Kaggle_AllState/'
        ensemble_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/ensemble/'
        feats_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/'
 
    
    #### Load
    train = pickle.load(open(feats_dir + 'train_model_5.pkl', 'rb') )
    test = pickle.load(open(feats_dir + 'test_model_5.pkl', 'rb') )

    n_train = train.shape[0]
    n_test = test.shape[0]
    
    print('\nMedian Loss:', train.loss.median())
    print('Mean Loss:', train.loss.mean())

    ids = pd.read_csv(directory + 'test.csv')['id']
    train_y = np.log(train['loss'] + shift)
    train_x = train.drop(['loss','id'], axis=1)
    test_x = test.drop(['loss','id'], axis=1)
    
    def objective(space):

        n_folds = 10
        cv_sum = 0
        #early_stopping = 100
        #fpred = []
        xgb_rounds = []
        
        kf = KFold(n_train, n_folds=n_folds)
        for i, (train_index, test_index) in enumerate(kf):
            print('\n Fold %d' % (i+1))
            X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
            y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]
        
            d_train = xgb.DMatrix(X_train, label=y_train)
            d_valid = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]

            clf = xgb.train(space,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=50,
                        obj=fair_obj,
                        feval=xg_eval_mae)

            xgb_rounds.append(clf.best_iteration)
        
            scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
            cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
            print('eval-MAE: %.6f' % cv_score)
            cv_sum = cv_sum + cv_score
        
        score = cv_sum / n_folds    
        return { 'loss': score, 'status': STATUS_OK }

    
    
    space = {
        'seed': 0,
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'silent': 1,
        'subsample': hp.uniform ('subsample', 0.6, 0.9),        
        'learning_rate' : hp.quniform('learning_rate', 0.02, 0.5, 0.025),
        'objective': 'reg:linear',
        'max_depth': hp.choice('max_depth', np.arange(10, 30, 5, dtype=int)),
        'min_child_weight': hp.choice('min_child', np.arange(20, 150, 15, dtype=int)), #20, 150, 20),
        'booster': 'gbtree'
    }

    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=1000, # change
                trials=trials)
    
    print(best)
    pickle.dumps(trials, open(directory + "hyperopt_trials.pkl", "wb") )
    #trials.to_pickle(directory + 'hyperopt_trials.pkl')
    #pickle.dumps