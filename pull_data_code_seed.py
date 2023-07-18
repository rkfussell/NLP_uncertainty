# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd


def get_data_train_test(train_dec, test_dec, code, val, s, n_full, n, train_size, test_institution, num_samples = 100):
    """
    Driver to run data collection processes in get_systematics_data.py

    Parameters
    ----------
    train_dec : float
        Decimal between 0 and 1 - portion of training data set to val
    test_dec : float
        Decimal between 0 and 1 - portion of full test data set to val (test sets sampled from full test set)
    s : int
        Seed number for current random instance
    code : string
        column in human-coded spreadsheet with the code of interest (e.g. "Expected" or "PLO")
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1)
    n_full: int
        number of responses in the full test set 
    n : int
        for each test, a sample of size n is pulled from the full test set
    train_size: int
        number of responses in the training set
    split_test : string
        denotes if data are split based on metadata to test population systematics
    opt_trustworthy : bool
        True if working with Trustworthy data only

    Returns
    -------
    df : Pandas Dataframe
        Dataframe of all data from current collection, can be concatenated with other df with other parameters

    """
    df = gsd.get_data(train_dec, test_dec, code, val, s, n_full, n, train_size, test_institution, num_samples = 100)
    df["train_size"] = train_size
    df["train_prop"] = train_dec
    df["test_prop"] = test_dec
    df["n_full"] = n_full
    df["n"] = n
    return df

def get_data_code_seed(code, val, s, filename, test_institution,  train_sizes = [600], train_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], test_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], n_fulls = [200], ns = [50,100], num_samples = 100):
    """
    Drive get_data_train_test() over a specified range of train_dec and test_dec
    
    Add all this data to a single csv.

    Parameters
    ----------
    code : string
        column in human-coded spreadsheet with the code of interest (e.g. "Expected" or "PLO")
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data (trustworthy))
    s : int
         Seed number for current random instance
    filename: string 
        phrase all files in this batch will start with
    train_sizes : list of ints
        list of train_size values, train_size is number of responses in the training set 
    train_decs : list of floats
        list of train_dec values, train_dec is number between 0 and 1 specifying fraction of responses with the code to put in training set
    test_decs : list of floats
        list of test_dec values, test_dec is number between 0 and 1 specifying fraction of responses with the code to put in test set
    n_fulls : list of ints
        list of n_full values, n_full is number of responses in the full test set 
    ns : list of ints
        list of n values, n is a sample of size n pulled from the full test set for each individual test
    split_test : string
        denotes if data are split based on metadata to test population systematics
    opt_trustworthy : bool
        True if working with Trustworthy data only

    Returns
    -------
    df : Pandas Dataframe
    
    train_dec : float
        Number between 0 and 1 specifying fraction of responses with the code to put in training set.

    """
    df = pd.DataFrame()
    #for testing on coded test and train
    for train_size in train_sizes:
        for train_dec in train_decs:
            for test_dec in test_decs:
                for n_full in n_fulls:
                    for n in ns:
                        if n < n_full:
                            df_new = get_data_train_test(train_dec, test_dec, code, val, s, n_full, n, train_size, test_institution)
                            if not df_new.empty:
                                df = pd.concat([df, df_new])
    #df.to_csv("male_vs_gender-min_" + split_test + "_code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
    df.to_csv(filename + test_institution + "_code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
    return df

