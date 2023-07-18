# -*- coding: utf-8 -*-
"""

@author: -
"""
import pandas as pd
import pull_data_code_seed as pull_data

def tests_on_inputs(df_s, Train_X, Train_y, full_test_X, full_test_y, val):
    #TEST 1: Train x and train y are same length, test x and test y are same length 
    assert(len(Train_X) == len(Train_y))
    assert(len(full_test_X) == len(full_test_y))
    #assert(len(full_test_X)> 125)
        
    #TEST 2: no empty or one character responses in the input data
    for df in df_s:
        assert(len(df[df["explanation"].str.len()<2]) == 0)

    #TEST 3: no overlap between data in train and test sets 
    #X_merge = pd.merge(pd.DataFrame(Train_X), pd.DataFrame(full_test_X))
    #print(X_merge)
    #assert(len(X_merge.index) == 0)

    
def tests_on_outputs(est, human_est,FP, FN, df_human, test_dec, train_dec, n_full, n):
    #TEST 1: Assert Human estimate = TP + FN; est is within shooting distance of human est. 
    for i, row in est.iteritems():
        assert(human_est[i] == FN[i] + est[i] - FP[i])
        #check TP + FP is in shooting distance of human estimate
        assert(est[i] - human_est[i] <15 or human_est[i]-est[i] <15)
        
    #TEST 2: No NAs in exported data for estimates – NAs are appropriately dealt with
    assert(not est.isnull().values.any())
    assert(not human_est.isnull().values.any())
    assert(not FP.isnull().values.any())
    assert(not FN.isnull().values.any())
    
    #TEST 3: human est out of 200 from df human is close to average of human_est in samples of 100
    assert(df_human["test_y"][1]/2 - human_est.mean() < 2, df_human["test_y"][1]/2 - human_est.mean())
    #TEST 4: human est is at right level given test_dec
    assert(100*test_dec - sum(human_est)/len(human_est) <3 or sum(human_est)/len(human_est) - 100*test_dec <3)
    #TEST 5: est must be less than n
    assert(est[i]<=n)
    
    
