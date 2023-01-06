# -*- coding: utf-8 -*-
"""

@author: -
"""



def tests_on_inputs(Train_X, Train_y, full_test_X, full_test_y, full_test_X_PRE, full_test_y_PRE, opt_trustworthy = TRUE):
    #TEST 1: Train x and train y are same length, test x and test y are same length 
    assert(len(Train_X) == len(Train_y))
    assert(len(full_test_X) == len(full_test_y))
    assert(len(full_test_X_PRE) == len(full_test_y_PRE))
    #can add an option to assert that each are an exact length, e.g. train = 600 and test = 200)
    if opt_trustworthy:
        assert(len(Train_X) == 600)
        assert(len(full_test_X) == 200)
        
    #TEST 2: no empty or one character responses in the input data
    #can do this with Train X
    
    #TEST 3: trials with same random seed are deterministically identical
    
    #TEST 4: no overlap between data in train and test sets
    X_merge = pd.merge(Train_X, full_test_X)
    assert(len(X_merge.index) == 0)
    
def tests_on_outputs(est, human_est,FP, FN):
    #TEST 1: Assert Human estimate = TP + FN; TP + FP is within 15% of human estimate than TN + FN. 
    assert(human_est = FN + est - FP)
    assert(est - human_est <15 or human_est-est <15)
    #TEST 2: No NAs in exported data for estimates – NAs are appropriately dealt with
    assert(!est.isnull().values.any())
    assert(!human_est.isnull().values.any())
    assert(!FP.isnull().values.any())
    assert(!FN.isnull().values.any())