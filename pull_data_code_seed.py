# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd

def get_data_train_test(train_dec,test_dec, s, code, val, opt_trustworthy):
    df = gsd.get_data(train_dec, test_dec, code, val, s, opt_trustworthy)
    df["train_prop"] = train_dec
    df["test_prop"] = test_dec
    return df

def get_data_code_seed(code, val, s, opt_trustworthy = True):
    train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    df = pd.DataFrame()
    for train_dec in train_decs:
        for test_dec in test_decs:
            df_new = get_data_train_test(train_dec, test_dec, s, code, val, opt_trustworthy)
            if not df_new.empty:
                df = pd.concat([df, df_new])
    df.to_csv("code" + code + "val" + str(val) + "seed" + str(s) + "_df.csv")

get_data_code_seed("PLO", "L", 1, opt_trustworthy = False)
get_data_code_seed("PLO", "P", 1, opt_trustworthy = False)
get_data_code_seed("PLO", "L", 2, opt_trustworthy = False)
get_data_code_seed("PLO", "P", 2, opt_trustworthy = False)
get_data_code_seed("PLO", "L", 3, opt_trustworthy = False)
get_data_code_seed("PLO", "P", 3, opt_trustworthy = False)


get_data_code_seed("Expected", 1, 1, opt_trustworthy = True)
get_data_code_seed("Consistent Results", 1, 1, opt_trustworthy = True)
get_data_code_seed("Good Methods", 1, 1, opt_trustworthy = True)
get_data_code_seed("Statistics", 1, 1, opt_trustworthy = True)

get_data_code_seed("Expected",1, 2, opt_trustworthy = True)
get_data_code_seed("Consistent Results", 1, 2, opt_trustworthy = True)
get_data_code_seed("Good Methods",1, 2, opt_trustworthy = True)
get_data_code_seed("Statistics",1, 2, opt_trustworthy = True)

get_data_code_seed("Expected",1, 3, opt_trustworthy = True)
get_data_code_seed("Consistent Results",1,3, opt_trustworthy = True)
get_data_code_seed("Good Methods",1, 3, opt_trustworthy = True)
get_data_code_seed("Statistics",1, 3, opt_trustworthy = True)


#TEST 5: trials with same random seed are deterministically identical