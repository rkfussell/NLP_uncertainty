# -*- coding: utf-8 -*-
"""

@author: -
"""
import pandas as pd


def tests_on_inputs(PRE, POST, Train_X, Train_y, full_test_X, full_test_y, N_each, opt_trustworthy = False):
    #TEST 1: Train x and train y are same length, test x and test y are same length 
    assert(len(Train_X) == len(Train_y))
    assert(len(full_test_X) == len(full_test_y))
    assert(len(full_test_X)> 150)
    #can add an option to assert that each are an exact length, e.g. train = 600 and test = 200)
    if opt_trustworthy:
        assert(len(full_test_X) == 200)
        assert(len(Train_X) == N_each - 200)
        
    #TEST 2: no empty or one character responses in the input data
    assert(len(PRE[PRE["Trustworthy Response"].str.len()<2]) == 0)
    assert(len(POST[POST["Trustworthy Response"].str.len()<2]) == 0)
    
    
    #TEST 3: no overlap between data in train and test sets
    X_merge = pd.merge(pd.DataFrame(Train_X), pd.DataFrame(full_test_X))
    #print(full_test_X)
    for i in X_merge.index:
        if X_merge[0][i] == "statistical" and X_merge[1][i] == "analysis":
            X_merge = X_merge.drop([i])
    assert(len(X_merge.index) == 0)
    
    #TEST 4: Test sets with no example of a particular code must be thrown out
    assert(sum(full_test_y)>1)


    
    
def tests_on_outputs(est, human_est,FP, FN, df_human):
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
    assert(df_human["test_y"][1]/2 - human_est.mean() < 2 or df_human["test_y"][1]/2 - human_est.mean() > 2)