# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd


def get_data_train_test(train_dec,test_dec, s, code, val, full_test_max, N_test, opt_trustworthy):
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
    opt_trustworthy : bool
        True if working with Trustworthy data only

    Returns
    -------
    df : Pandas Dataframe
        Dataframe of all data from current collection, can be concatenated with other df with other parameters

    """
    df = gsd.get_data(train_dec, test_dec, code, val, s, full_test_max, N_test, opt_trustworthy)
    df["train_prop"] = train_dec
    df["test_prop"] = test_dec
    df["full_test_max"] = full_test_max
    df["N_test"] = N_test
    return df

def get_data_code_seed(code, val, s, opt_trustworthy = True):
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
    opt_trustworthy : bool
        True if working with Trustworthy data only

    Returns
    -------
    df : Pandas Dataframe

    """
    #full_test_maxs = [200]
    #N_tests = [100]
    train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    full_test_maxs = [120,130,140,150,160,170,180,190,200,225,250,275,300,325,350,375,400,425,450,475,500]
    N_tests = [5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    full_test_maxs = [200]
    df = pd.DataFrame()
    for train_dec in train_decs:
        for test_dec in test_decs:
            for full_test_max in full_test_maxs:
                for N_test in N_tests:
                    if N_test < full_test_max:
                        df_new = get_data_train_test(train_dec, test_dec, s, code, val, full_test_max, N_test, opt_trustworthy)
                        if not df_new.empty:
                            df = pd.concat([df, df_new])
    df.to_csv("BIG_test_vary_" + "code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
    return df

