#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:46:03 2018

@author: sadams
"""

from sklearn.metrics import confusion_matrix

def scania_score(y_true,y_pred):
    '''
    Calculates Scania's custom score for APS classification problem in heavy trucks.
    Type 2 errors are more heavily penalized, and a lower score is better.
    
    Score is 10*(no. of false positives) + 500*(no. of false negatives)
    Type 1 error represents a mechanic performing an unnecessary inspection.
    Type 2 error represents a failure to identify a faulty truck, possibly resulting in a break-down.
    ---
    Inputs:
    y_true,y_pred: array-like objects of equal length representing true and predicted labels.
    
    Returns:
    float value (>= 0) representing cost of Type 1 and 2 errors in classification.
    '''
    confmat = confusion_matrix(y_true, y_pred)
    
    return confmat[0,1]*10 + confmat[1,0]*500