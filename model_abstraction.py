#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:34:53 2018

@author: sadams
"""

from collections import defaultdict
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import xgboost as xgb

def cross_val_models(models, X, y, use_cv=5, params=defaultdict(dict), metric = 'roc_auc', verbose = False):
    defaultdict
    '''
    Accepts dictionary of models to test, using default parameters unless otherwise specified.
    Models and parameters dictionaries must have matching keys
    Currently assumes X has been scaled, normalized, or transformed as needed.
    '''
    results = defaultdict(str)
    
    for name, model in models.items():
        cv_score = np.mean(cross_val_score(model(**params[name]),X,y,cv=use_cv, scoring = metric))
        
        if verbose:
            print('Model:', name, 'Metric:', metric, cv_score)
                  
        results[name] = cv_score
    
    return results

def cross_val_xgb(X,y, folds, cv_scorer, pred_threshold=0.5, fit_metric='auc',
                  model_objective='binary:logistic'):
    '''
    Performs cross-validation on an XGBoost estimator object. Fits a model on
    each fold of the provided data.
    Returns cross-validation error measurements.
    '''
    
    def prob_to_pred(num, cutoff=pred_threshold):
        # Converts xgb prediction output to binary value
        return 1 if num > cutoff else 0
    
    # Prepare to store individual fold scores
    cv_scores = []
    
    # Fit model for each fold and retain error metrics
    for train_idx, val_idx in folds.split(X, y):
        
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
       
        gbm = xgb.XGBRegressor( 
                           n_estimators=30000, #arbitrary large number b/c we're using early stopping
                           max_depth=3,
                           objective= model_objective,
                           learning_rate=.1, 
                           subsample=1,
                           min_child_weight=1,
                           colsample_bytree=.8
                          )
        eval_set=[(X_tr,y_tr),(X_val,y_val)] #tracking train/validation error as we go
        fit_model = gbm.fit( 
                        X_tr, y_tr, 
                        eval_set=eval_set,
                        eval_metric=fit_metric,
                        early_stopping_rounds=50, # stop when validation error hasn't improved in this many rounds
                        verbose=False #gives output log as below
                       )
        # Make and assess validation predicitons
        y_pred = pd.Series(fit_model.predict(X_val,
            ntree_limit=gbm.best_ntree_limit)).apply(prob_to_pred)
        
        cv_scores.append(cv_scorer(y_val,y_pred))
    
    # Calculate CV Error and mean custom error if applicable
    return np.mean(cv_scores)
        