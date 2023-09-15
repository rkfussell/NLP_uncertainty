# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd


def get_data_train_test(p_train,p_test, s, code, val, N_bank, n, train_size, split_test, opt_trustworthy, num_samples):
    """
    Driver to run data collection processes in get_systematics_data.py

    Parameters
    ----------
    p_train : float
        Decimal between 0 and 1 - portion of training data set to val
    p_test : float
        Decimal between 0 and 1 - portion of full test data set to val (test sets sampled from full test set)
    s : int
        Seed number for current random instance
    code : string
        column in human-coded spreadsheet with the code of interest (e.g. "Expected" or "PLO")
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1)
    N_bank: int
        number of responses in the full test set 
    n : int
        for each test, a sample of size n is pulled from the full test set
    train_size: int
        number of responses in the training set
    split_test : string
        denotes if data are split based on metadata to test population systematics
    opt_trustworthy : bool
        True if working with Trustworthy data only
    num_samples: int
        number of times a sample of size n will be pulled from the test bank of size N_bank

    Returns
    -------
    df : Pandas Dataframe
        Dataframe of all data from current collection, can be concatenated with other df with other parameters

    """
    df = gsd.get_data(p_train, p_test, code, val, s, N_bank, n, train_size, split_test, opt_trustworthy, num_samples)
    df["train_size"] = train_size
    df["train_prop"] = p_train
    df["test_prop"] = p_test
    df["N_bank"] = N_bank
    df["n"] = n
    return df

def get_data_code_seed(code, val, s, filename, train_sizes = [600], p_trains = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], p_tests = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], N_banks = [200], ns = [50,100], split_test = "all", opt_trustworthy = True, num_samples = 100):
    """
    Drive get_data_train_test() over a specified range of p_train and p_test
    
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
    p_trains : list of floats
        list of p_train values, p_train is number between 0 and 1 specifying fraction of responses with the code to put in training set
    p_tests : list of floats
        list of p_test values, p_test is number between 0 and 1 specifying fraction of responses with the code to put in test set
    N_banks : list of ints
        list of N_bank values, N_bank is number of responses in the full test set 
    ns : list of ints
        list of n values, n is a sample of size n pulled from the full test set for each individual test
    split_test : string
        denotes if data are split based on metadata to test population systematics
    opt_trustworthy : bool
        True if working with Trustworthy data only
    num_samples: int
        number of times a sample of size n will be pulled from the test bank of size N_bank

    Returns
    -------
    df : Pandas Dataframe
    
    p_train : float
        Number between 0 and 1 specifying fraction of responses with the code to put in training set.

    """
    df = pd.DataFrame()
    #for gathering labels on uncoded data
    if split_test == "F22":
        df = gsd.get_data(0.5, 0.5, code, val, s, 200, 100, train_sizes[0], split_test, opt_trustworthy, num_samples)
        df.to_csv("coded_F22_data_" + code + ".csv")
    else:
        #for testing on coded test and train
        for train_size in train_sizes:
            for p_train in p_trains:
                for p_test in p_tests:
                    for N_bank in N_banks:
                        for n in ns:
                            if n < N_bank:
                                df_new = get_data_train_test(p_train, p_test, s, code, val, N_bank, n, train_size, split_test, opt_trustworthy, num_samples)
                                if not df_new.empty:
                                    df = pd.concat([df, df_new])
        #df.to_csv("male_vs_gender-min_" + split_test + "_code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
        df.to_csv(filename + split_test + "_code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")
    return df

