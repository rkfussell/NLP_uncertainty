# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd


def get_data_train_test(train_dec,test_dec, s, code, val, opt_trustworthy):
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
    df = gsd.get_data(train_dec, test_dec, code, val, s, opt_trustworthy)
    df["train_prop"] = train_dec
    df["test_prop"] = test_dec
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
    train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    df = pd.DataFrame()
    for train_dec in train_decs:
        for test_dec in test_decs:
            df_new = get_data_train_test(train_dec, test_dec, s, code, val, opt_trustworthy)
            if not df_new.empty:
                df = pd.concat([df, df_new])
    df.to_csv("code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
    return df

