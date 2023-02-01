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
    """
    Sets seed in numpy and random

    Parameters
    ----------
    seed : int
         Seed number for current random instance.

    Returns
    -------
    None.

    """
    np.random.seed(seed)
    random.seed(seed)
def preprocess_text(line):
    """
    Preprocess responses, split into tokens. 
    
    Make all characters lowercase, remove punctuation and numbers, remove single characters, remove multiple spaces,
    ensure all words are alphabetic

    Parameters
    ----------
    line : string
        DESCRIPTION.

    Returns
    -------
    tokens : list
        clearned, tokenized list of words

    """
    line = line.lower()
    # Remove punctuations and numbers
    line = re.sub('[^a-zA-Z]', ' ', line)
    # Single character removal
    line = re.sub(r"\s+[a-zA-Z]\s+", ' ', line)
    # Removing multiple spaces
    line = re.sub(r'\s+', ' ', line)
    
    tokens = line.split()
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

def cohens_kappa(tp,fp,fn,tn):
    """
    Calculate Cohen's kappa

    Parameters
    ----------
    tp : int
        True Positive
    fp : int
        False Positive
    fn : int
        False Negative
    tn : int
        True Negative

    Returns
    -------
    float
        Cohen's kappa, between 0 and 1

    """
    #assert(isinstance(tp,int))
    #assert(isinstance(fp,int))
    #assert(isinstance(fn,int))
    #assert(isinstance(tn,int))
    num = 2*(tp*tn - fn*fp)
    denom = (tp+fp)*(fp+tn) + (tp+fn)*(fn+tn)
    return num/denom

def logreg1(s,tokenizer, Xtrain, Xtest, Train_y, Test_y):
    """
    Run logistic regression machine learning algorithm 
    Parameters
    ----------
    s : integer
        s is an integer specifying random seed.
    tokenizer : Tokenizer object
        Stores words in vocabulary wotj relevant index (necessary for viewing log reg coefficients)
    Xtrain : numpy array
        X (encoded responses) of training set
    Xtest : numpy array
         X (encoded responses) of test set
    Train_y : numpy array
        y (classification) of training set
    Test_y : TYPE
        y (encoded responses) of test set

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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

def get_stats_est_fp_fn_binary(train_x, train_y, test_x, test_y, trials = 100):
    """
    Run the logistic regression for many trials, store and return all relevant data (e.g. computer est, human est, rate of false negatives and rate of false positives)
    
    For binary classification tasks. 
    
    Parameters
    ----------
    train_x : numpy array
        X (encoded responses) of training set, all training data used in all trials
    train_y : numpy array
        y (classification) of training set, all training data used in all trials
    test_x : numpy array
        X (encoded responses) of test set, random samples of 100 pulled from here in all trials
    test_y : numpy array
        X (encoded responses) of test set, random samples of 100 pulled from here in all trials
    trials : int, optional
        Number of random samples to pull from the test sets to get multiple trials from the training set. The default is 100.


    Returns
    -------
    df: pandas_dataframe 
        Dataframe of all trials (default 100) for given parameters. 
    df_human: pandas dataframe
        Value counts for each code, for testing purposes

    """
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

def get_stats_est_fp_fn_3code(train_x, train_y, test_x, test_y, val, trials = 100):
    """
    Run the logistic regression for many trials, store and return all relevant data (e.g. computer est, human est, rate of false negatives and rate of false positives)
    
    When classification task has 3 possible results. 
    
    Parameters
    ----------
    train_x : numpy array
        X (encoded responses) of training set, all training data used in all trials
    train_y : numpy array
        y (classification) of training set, all training data used in all trials
    test_x : numpy array
        X (encoded responses) of test set, random samples of 100 pulled from here in all trials
    test_y : numpy array
        X (encoded responses) of test set, random samples of 100 pulled from here in all trials
    trials : int, optional
        Number of random samples to pull from the test sets to get multiple trials from the training set. The default is 100.

    Returns
    -------
    df: pandas_dataframe 
        Dataframe of all trials (default 100) for given parameters. 
    df_human: pandas dataframe
        Value counts for each code, for testing purposes

    """
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
            vals = ["L","O","P"]
            if sum(1 for i in Te_y if i == vals[0])<2 or sum(1 for i in Te_y if i == vals[1])<2 or sum(1 for i in Te_y if i == vals[2])<2:
                n += 1
                print(n)
            else:
                enough_pos_in_test = True
                
            if n > 21:
                return pd.DataFrame(), pd.DataFrame()

        Te_X = tok.texts_to_matrix(Te_X, mode="binary")
        mat= logreg1(s,tok, Tr_X, Te_X, train_y, Te_y)
        #list of 3 codes in alphabetical order
        if val == val[0]:
            est.append(mat[0][0] + mat[1][0] + mat[2][0])
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[0])
            fp.append(mat[1][0] + mat[2][0])
            fn.append(mat[0][1] + mat[0][2])
            kappa.append(cohens_kappa(mat[0][0], fp[-1], fn[-1], mat[1][1]+ mat[2][2]))
        elif val == val[1]:
            est.append(mat[0][1] + mat[1][1] + mat[2][1])
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[1])
            fp.append(mat[0][1] + mat[2][1])
            fn.append(mat[1][0] + mat[1][2])
            kappa.append(cohens_kappa(mat[1][1], fp[-1], fn[-1], mat[0][0]+ mat[2][2]))
        elif val == val[2]:
            est.append(mat[0][2] + mat[1][2] + mat[2][2])
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[2])
            fp.append(mat[0][2] + mat[1][2])
            fn.append(mat[2][0] + mat[2][1])
            kappa.append(cohens_kappa(mat[2][2], fp[-1], fn[-1], mat[0][0]+ mat[1][1]))
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
def trustworthy_process( s, code):
    """
    Pre-process trustworthy data, break into data subsets PRE and POST

    Parameters
    ----------
    s : integer
        s is an integer specifying random seed.
    code : string
        the category of response in human coded data.

    Returns
    -------
    tuple of full data frame for each data subset (subsets defined by systematic variable of interest), tuple
        DESCRIPTION.
    N_each, int
        Description
    tuple of X for each data subset
        Description
    tuple of y for each data subset
        Description
    """
    #read in the data
    df = pd.read_csv(r'trustworthy_dat.csv')
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
    
    return (POST, PRE), N_each, (X_PRE, X_POST), (y_PRE, y_POST)

def sources_process( s, code ):
    """
    Pre-process sources of uncertainty data, break into groups of systematic variable.
    
    Data subsets as subsets defined by systematic variable of interest (exp): PM, BM, SG, SS. 

    Parameters
    ----------
    s : integer
        s is an integer specifying random seed.
    code : string
        the category of response in human coded data.

    Returns
    -------
    tuple of full data frame for each data subset (subsets defined by systematic variable of interest), tuple
        DESCRIPTION.
    N_each, int
        Description
    tuple of X for each data subset
        Description
    tuple of y for each data subset
        Description
    """
    #read in the data
    df = pd.read_csv(r'PLO_dat.csv')
    df = df[df["Q"].notnull()]
    df = df[df["Q"].str.len()>1]
    
    #Split by experiment, then shuffle all the rows randomly but deterministically
    PM = df[df["Exp"] =="PM"]
    BM = df[df["Exp"] =="BM"]
    SG = df[df["Exp"] =="SG"]
    SS = df[df["Exp"] =="SS"]
    set_random_seed(seed = s)
    PMs = PM.sample(frac=1).reset_index(drop=True)
    BMs = BM.sample(frac=1).reset_index(drop=True)
    SGs = SG.sample(frac=1).reset_index(drop=True)
    SSs = SS.sample(frac=1).reset_index(drop=True)

    #Pull out only the first 239 from each experiment (because min response number 239 for SS)
    PMs = PMs.iloc[:239]
    BMs = BMs.iloc[:239]
    SGs = SGs.iloc[:239]
    SSs = SSs.iloc[:239]
    #Keep the 468-239 additional PM responses because those will be useful
    PM_add = PMs.iloc[239:]
    BM_add = BMs.iloc[239:]


    N_each = len(PMs)
    assert(N_each*4 - 200 > 700)
    
    #pre process responses (X) and match to code value (y) 
    X_PMs = []
    for response in PMs["Q"].tolist():
        X_PMs.append(preprocess_text(response))
    y_PMs = np.array(PMs[code].tolist())
    
    X_BMs = []
    for response in BMs["Q"].tolist():
        X_BMs.append(preprocess_text(response))
    y_BMs = np.array(BMs[code].tolist())
    
    X_SGs = []
    for response in SGs["Q"].tolist():
        X_SGs.append(preprocess_text(response))
    y_SGs = np.array(SGs[code].tolist())
    
    X_SSs = []
    for response in SSs["Q"].tolist():
        X_SSs.append(preprocess_text(response))
    y_SSs = np.array(SSs[code].tolist())
    
    X_PM_add = []
    for response in PMs["Q"].tolist():
        X_PM_add.append(preprocess_text(response))
    y_PM_add = np.array(PMs[code].tolist())
    
    return (PMs, BMs, SGs, SSs), N_each, (X_PMs, X_BMs, X_SGs, X_SSs), (y_PMs, y_BMs, y_SGs, y_SSs)

def get_data(train_dec, test_dec, code, val, s, opt_trustworthy = False):
    """
    
    code is a string, the category of response in human coded data
    N_each is the number of responses after cleaning 

    Parameters
    ----------
    train_dec : float
        Number between 0 and 1 specifying fraction of PRE to put in training set.
    test_dec : float
        Number between 0 and 1 specifying fraction of PRE to put in test set.
    code : string
        the category of response in human coded data.
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data (trustworthy))
    s : integer
        s is an integer specifying random seed.
    opt_trustworthy : bool
        True if working with Trustworthy data only
    

    Returns
    -------
    df : Pandas dataframe
        Dataframe of all trials (default 100) for given parameters. 
        
        All data at this stage has been run through the test_bank. 

    """
    
    if opt_trustworthy:
        df_s, N_each, X_s, y_s =  trustworthy_process(s, code)
        full_test_max = 200 
        train_size = 600
    else:
        df_s, N_each, X_s, y_s = sources_process(s, code)
        full_test_max = 150 
        train_size = 600
    
    #number of pos and neg in test and training sets
    N_pos_test = math.floor(full_test_max*test_dec)
    N_neg_test = full_test_max - N_pos_test
    N_pos_train = math.floor(train_size*train_dec)
    N_neg_train = train_size - N_pos_train
    
    print("\n")
    print("N_pos_test " + str(N_pos_test))
    print("N_neg_test " + str(N_neg_test))
    print("N_pos_train " + str(N_pos_train))
    print("N_neg_train " + str(N_neg_train))
    
    X_pos = []
    X_neg = []
    y_pos = []
    y_neg = []
    #sub denotes the subgroups of a dataset, e.g. pre and post for trustworthy, experiment type for sources
    for sub_idx in range(len(X_s)):
        for idx, condition_met in enumerate(y_s[sub_idx] == val):
            if condition_met:
                X_pos.append(X_s[sub_idx][idx])
                y_pos.append(y_s[sub_idx][idx])
            elif not condition_met:
                X_neg.append(X_s[sub_idx][idx])
                y_neg.append(y_s[sub_idx][idx])

    """
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
            
    X_pos = X_PRE_pos + X_POST_pos
    X_neg = X_PRE_neg + X_POST_neg
    """
    
    assert(sum(1 for i in y_pos if i == val)/len(y_pos) == 1.0)
    assert(sum(1 for i in y_neg if i == val)/len(y_neg) == 0.0)
    
    random.shuffle(X_pos)
    random.shuffle(X_neg)
    
    print("length X_pos " + str(len(X_pos)))
    print("length X_neg " + str(len(X_neg)))
    
    if len(X_pos) < N_pos_train + N_pos_test or len(X_neg) < N_neg_train + N_neg_test:
        return pd.DataFrame()

    
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
    if opt_trustworthy:
        df, df_human = get_stats_est_fp_fn_binary(Train_X, Train_y, full_test_X, full_test_y)
    else:
        df, df_human = get_stats_est_fp_fn_3code(Train_X, Train_y, full_test_X, full_test_y, val)
    if df.empty:
        return pd.DataFrame()
    if sum(1 for i in Train_y if i == val)<2:
        warnings.warn("Less than 2 examples of code in Training set")
    test_bank.tests_on_inputs(df_s, Train_X, Train_y, full_test_X, full_test_y, N_each, val, opt_trustworthy)
    #might be able to not output the human dfs because of the tests, write down where that info can be gathered elsewhere
    test_bank.tests_on_outputs(df["est"], df["human_est"],df["fp"], df["fn"], df_human)
    return df
