# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""
import pandas as pd
import numpy as np
import re
import random
import test_bank
import warnings
import math

from sklearn import model_selection, naive_bayes, svm, linear_model, ensemble, neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer

def set_random_seed(seed=0):
  np.random.seed(seed)
  random.seed(seed)
def preprocess_text(para):
    para = para.lower()
    # Remove punctuations and numbers
    para = re.sub('[^a-zA-Z]', ' ', para)
    # Single character removal
    para = re.sub(r"\s+[a-zA-Z]\s+", ' ', para)
    # Removing multiple spaces
    para = re.sub(r'\s+', ' ', para)
    
    tokens = para.split()
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

def cohens_kappa(tp,fp,fn,tn):
    num = 2* (tp*tn - fn*fp)
    denom = (tp+fp)*(fp+tn) + (tp+fn)*(fn+tn)
    return num/denom

def logreg1(s,tokenizer, Xtrain, Xtest, Train_y, Test_y):
    set_random_seed(seed = s)
    Log = linear_model.LogisticRegression(random_state = s, max_iter = 10000)
    Log.fit(Xtrain,Train_y)
    # predict the labels on validation dataset
    predictions_Log = Log.predict(Xtest)
    #if s==0:
    if False:
        #Log.coef_[2][i] for principles, Log.coef_[0][i] for limitations
        coefs_dict = {i: Log.coef_[0][i] for i in range(len(Log.coef_[0]))}
        sorted_dict = {}
        sorted_keys = sorted(coefs_dict, key=coefs_dict.get) 
        words_dict = dict((v,k) for k,v in tokenizer.word_index.items())
        #for w in sorted_keys:
        #    sorted_dict[w] = coefs_dict[w]
        #print(sorted_dict)
        print("\nNegative:")
        for num in sorted_keys[:10]:
            print(words_dict[num])
        print("\nPositive:")
        for num in sorted_keys[-20:]:
            print(words_dict[num])
    return confusion_matrix(Test_y, predictions_Log)

def get_stats_est_fp_fn_trust(train_x, train_y, test_x, test_y, trials = 100):
    est = []
    fp = []
    fn = []
    human_est = []
    kappa = []
    for s in range(trials):
        set_random_seed(seed = s)
        tok = Tokenizer(lower = False)
        #tok.fit_on_texts(Train_X)
        tok.fit_on_texts(train_x)
        #Tr_X = tok.texts_to_matrix(Train_X, mode="binary")
        Tr_X = tok.texts_to_matrix(train_x, mode="binary")
        #make test set, must have at least 2 examples of the code
        enough_pos_in_test = False
        n = 0
        while not enough_pos_in_test:
            set_random_seed(seed = s + n*100)
            _, Te_X, _, Te_y = model_selection.train_test_split(test_x,test_y,test_size=100)
            if sum(Te_y)<2:
                n +=1
                print(n)
            else:
                enough_pos_in_test = True
                
            if n > 21:
                return pd.DataFrame(), pd.DataFrame()

        Te_X = tok.texts_to_matrix(Te_X, mode="binary")
        mat = logreg1(s,tok, Tr_X, Te_X, train_y, Te_y)
        
        est.append(mat[1][1] + mat[0][1])
        human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[1])
        fp.append(mat[0][1])
        fn.append(mat[1][0])
        kappa.append(cohens_kappa(mat[1][1], mat[1][0], mat[0][1], mat[0][0]))
        #except:
        #    est.append(0)
        #    human_est.append(pd.DataFrame(Te_y).value_counts(sort = False))
        #    fp.append(0)
        #    fn.append(0)
        #    kappa.append(-1)

    df = pd.DataFrame({
        "est" : est,
        "fp" : fp,
        "fn" : fn,
        "kappa" : kappa,
        "human_est": human_est
    })
    df_human = pd.DataFrame({
        "train_y": pd.DataFrame(train_y).value_counts(sort = False),
        "test_y": pd.DataFrame(test_y).value_counts(sort = False)})
    
    return df, df_human


def get_data_trust(train_dec, test_dec, s = 1, code = "Expected", ):
    """
    
    code is a string, the category of response in human coded data
    N_each is the number of responses after cleaning 

    Parameters
    ----------
    s : integer, optional
        s is an integer specifying random seed. The default is 1.
    code : string
        the category of response in human coded data.
    N_each : integer, optional
        N_each is the max number of responses from either type (PRE or POST) after cleaning. 
    train_perc : float
        Number between 0 and 1 specifying fraction of PRE to put in training set.
    test_perc : float
        Number between 0 and 1 specifying fraction of PRE to put in test set.

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    df_human : TYPE
        DESCRIPTION.
    df_PRE : TYPE
        DESCRIPTION.
    df_human_PRE : TYPE
        DESCRIPTION.

    """
    #read in the data
    df = pd.read_excel (r'Trustworthy_Master_Spreadsheet_Summer_2022.xlsx')
    df["Trustworthy Response"] = df["Trustworthy Response"].str.replace(".","")
    df = df[df["Trustworthy Response"].notnull()]
    df = df[df["Trustworthy Response"].str.len()>1]
    #remove duplicates from master spreadsheet
    df = df[~df.duplicated(subset = "Trustworthy Response", keep = "first")]
    df = df[df["ResponseID"]!=124]
    df = df[df["ResponseID"]!=308]
    #create equal sized, shuffled pre and post data frames. these are fixed and do not depend on seed.
    POST = df[df["PRE/POST"] == "POST"]
    PRE = df[df["PRE/POST"] == "PRE"]
    
    set_random_seed(seed = s)
    POST = POST.sample(frac=1).reset_index(drop=True)
    PRE = PRE.sample(frac=1).reset_index(drop=True)
    print(len(PRE))
    print(len(POST))
    N_each = len(PRE) if len(PRE) < len(POST) else len(POST)
    assert(N_each > 800)
    
    #keep N_each from both PRE and POST because there are only N_each in PRE (same every time, fixed at seed 132)
    PRE = PRE[:N_each]
    POST = POST[:N_each]
    
    #shuffle PRE and POST data frames according to current seed
    #set_random_seed(seed = s)
    POST = POST.sample(frac=1).reset_index(drop=True)
    PRE = PRE.sample(frac=1).reset_index(drop=True)
    #pre process responses (X) and match to code value (y) 
    X_PRE = []
    for response in PRE["Trustworthy Response"].tolist():
        X_PRE.append(preprocess_text(response))
    y_PRE = np.array(PRE[code].tolist())
    X_POST = []
    for response in POST["Trustworthy Response"].tolist():
        X_POST.append(preprocess_text(response))
    y_POST = np.array(POST[code].tolist())
    
    #cook percentage of code in training set
    overall_frequency = sum(y_PRE + y_POST)
    
    #number of pos and neg in test and training sets
    N_pos_test = math.floor(200*test_dec)
    N_neg_test = 200 - N_pos_test
    N_pos_train = math.floor(800*train_dec)
    N_neg_train = 800 - N_pos_train
    
    print("N_pos_test " + str(N_pos_test))
    print("N_neg_test " + str(N_neg_test))
    print("N_pos_train " + str(N_pos_train))
    print("N_neg_train " + str(N_neg_train))
    
    
    X_PRE_pos = []
    X_PRE_neg = []
    X_POST_pos = []
    X_POST_neg = []
    y_PRE_pos = []
    y_PRE_neg = []
    y_POST_pos = []
    y_POST_neg = []
    
    for idx, condition_met in enumerate(y_PRE == 1):
        if condition_met:
            X_PRE_pos.append(X_PRE[idx])
            y_PRE_pos.append(y_PRE[idx])
        elif not condition_met:
            X_PRE_neg.append(X_PRE[idx])
            y_PRE_neg.append(y_PRE[idx])
    for idx, condition_met in enumerate(y_POST == 1):
        if condition_met:
            X_POST_pos.append(X_POST[idx])
            y_POST_pos.append(y_POST[idx])
        elif not condition_met:
            X_POST_neg.append(X_POST[idx])
            y_POST_neg.append(y_POST[idx])
    
    assert(sum(y_PRE_pos)/len(y_PRE_pos) == 1.0)
    assert(sum(y_PRE_neg)/len(y_PRE_neg) == 0.0)
    assert(sum(y_POST_pos)/len(y_POST_pos) == 1.0)
    assert(sum(y_POST_neg)/len(y_POST_neg) == 0.0)
    
    X_pos = X_PRE_pos + X_POST_pos
    X_neg = X_PRE_neg + X_POST_neg
    random.shuffle(X_pos)
    random.shuffle(X_neg)
    
    if len(X_pos) < N_pos_train + N_pos_test or len(X_neg) < N_neg_train + N_neg_test:
        return pd.DataFrame()
    
    y_pos = y_PRE_pos + y_POST_pos
    y_neg = y_PRE_neg + y_POST_neg
    print("length X_pos " + str(len(X_pos)))
    print("length X_neg " + str(len(X_neg)))
    #create X and y for fixed training set
    Train_X = X_pos[:N_pos_train] + X_neg[:N_neg_train]
    Train_y = np.concatenate((y_pos[:N_pos_train],y_neg[:N_neg_train]))
    
    #create a test set that is sampled later
    full_test_X = X_pos[N_pos_train:N_pos_train + N_pos_test] + X_neg[N_neg_train: N_neg_train + N_neg_test]
    full_test_y = np.concatenate((y_pos[N_pos_train:N_pos_train + N_pos_test], y_neg[N_neg_train: N_neg_train + N_neg_test]))
    
    
    print("train X " + str(len(Train_X)))
    print("train y " + str(len(Train_y)))
    print("test X " + str(len(full_test_X)))
    print("test y " + str(len(full_test_y)))
    
    if len(full_test_y) <= 150:
        return pd.DataFrame()
    """
    #create train set size based on input params for percent PRE
    N_PRE_train = math.floor((N_each-200)*train_dec)
    N_POST_train = N_each - 200 - N_PRE_train
    
    #create X and y for fixed training set
    Train_X = X_PRE[:N_PRE_train] + X_POST[:N_POST_train]
    Train_y = np.concatenate((y_PRE[:N_PRE_train],y_POST[:N_POST_train]))
    
    #create test set size based on input params
    N_PRE_test = math.floor(200*test_dec)
    print(N_PRE_test)
    N_POST_test = 200 - N_PRE_test
    print(N_POST_test)
    #create a test set that is sampled later
    full_test_X = X_PRE[N_PRE_train:N_PRE_train + N_PRE_test] + X_POST[N_POST_train: N_POST_train + N_POST_test]
    full_test_y = np.concatenate((y_PRE[N_PRE_train:N_PRE_train + N_PRE_test], y_POST[N_POST_train: N_POST_train + N_POST_test])) 
    """
    
    #Run logreg on 100 test set samples (equal test set)
    df, df_human = get_stats_est_fp_fn_trust(Train_X, Train_y, full_test_X, full_test_y)
    if df.empty:
        return pd.DataFrame()
    if sum(Train_y)<2:
        warnings.warn("Less than 2 examples of code in Training set")
    test_bank.tests_on_inputs(PRE, POST, Train_X, Train_y, full_test_X, full_test_y, N_each, opt_trustworthy = False)
    #might be able to not output the human dfs because of the tests, write down where that info can be gathered elsewhere
    test_bank.tests_on_outputs(df["est"], df["human_est"],df["fp"], df["fn"], df_human)
    return df
