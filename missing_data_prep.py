#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:19:19 2018

@author: sadams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute,\
    IterativeImputer, BiScaler

def ready_aps_data(dropna=False, drop_threshold = 0.20):
    """
    Prepares training and testing sets for APS failure classification project.
    Returns pandas dataframes with all missing measurments imputed.
        X_train: pandas dataframe, training set features
        X_test: pandas dataframe, testing set labels
        y_train: pandas series, training set labels
        y_test pandas series, testing set lables
    
    """
    ''' Make this the fxn input, rather than hardcoded, so you can reuse it for other stuff.
    Add more defensive programming: error checks, data not useable, etc.'''
    folder = 'aps-failure-at-scania-trucks-data-set/'
    test_csv = 'aps_failure_test_set.csv'
    train_csv = 'aps_failure_training_set.csv'
    
    training_data = pd.read_csv(folder+train_csv)
    testing_data = pd.read_csv(folder+test_csv)
    
    # Convert labels and data to numeric values
    training_data = prepare_dataframe(training_data)
    testing_data = prepare_dataframe(testing_data)
    
    # Drop columns with mostly missing values
    high_nan_cols = get_nan_frac_cols(training_data,
        cutoff=drop_threshold,
        graph=True)
    
    return training_data.iloc[:,1:], testing_data.iloc[:,1:],\
        training_data['class'].astype(int), testing_data['class'].astype(int)

def prepare_dataframe(df):
    '''
    Accepts a pandas dataframe with missing values as 'na' and numbers as string objects.
    Returns a pandas dataframe with all missing values replaced.
    Adds feature identifying how much missing data is present.
    '''
    # Dealing with missing data labeled as 'na'
    df.replace('na', np.nan, inplace=True)
    df['class'] = df['class'].apply(class_to_numeric, convert_dtype = True)\
        .astype(int)
    
    # Converting feature columns to numeric values
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors = 'raise')
    
    # Creating new feature: count of missing data in an example
    n_features = df.shape[1]
    df['n_missing'] = n_features - df.count(axis=1)
    
    return df

def class_to_numeric(label):
    """
    Converts target labels from strings to 1 or 0 for binary classification.
    """
    if label == 'pos':
        return 1
    elif label == 'neg':
        return 0
    else:
        raise ValueError('Unexpected target label: '+label)


def get_nan_frac_cols(df, cutoff = 0.20, graph=False):
    '''
    Returns column labels for columns with a NaN values composition >= a cutoff fraction.
    Intended use is to visualize how prevalent NaN values are in a dataset.
    ---
    Parameters
    df: Pandas dataframe. Each column represents a feature and its data for all observations.
    cutoff: decimal value, the acceptable fraction of data that can be NaN
    graph: bool, plots the data if True
    
    Returns
    List of dataframe column labels that have NaN fraction > cutoff fraction
    '''
    nan_frac_vec, cols_high_nan = [], []
    
    for column in df.columns:
        nan_frac = df[column][df[column].isnull()].size/df.shape[0]
        nan_frac_vec.append(nan_frac)

        if nan_frac >= cutoff:
            cols_high_nan.append(column)
    
    # Visualize the amount of missing data: features on x-axis, fraction of
    # feature values missing on y axis. Threshold line plotted for emphasis.       
    if graph:
        sns.scatterplot(range(df.shape[1]),nan_frac_vec, alpha = 0.9, label='data')
        plt.plot(range(df.shape[1]), np.ones(df.shape[1])*cutoff,'r-', label='Cutoff Threshold')
        plt.legend(loc='best')
        plt.title('Data by column with {}% cutoff'.format(100*cutoff))
        plt.ylabel('Fraction of Values that are NaN')
        plt.xlabel('Column position')
    
    return cols_high_nan


def reject_outliers(data, m=5):
    '''
    Accepts a 2D Numpy array.
    Returns the indices rows with all values less than m times larger than the mean for that column.
    Effectively: returns indices for rows without outlier data.
    '''
    array_bools = abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0)
    indexes_not_outliers = np.apply_along_axis(all,1,array_bools)
    return indexes_not_outliers