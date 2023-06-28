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

import contractions
import nltk

from sklearn import model_selection, naive_bayes, svm, linear_model, ensemble, neighbors
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
    
    Pre-processing help from: https://www.kaggle.com/code/frankmollard/nlp-a-gentle-introduction-lstm-word2vec-bert

    Parameters
    ----------
    line : string
        unprocessed chunk of text (single response)

    Returns
    -------
    tokens : list
        clearned, tokenized list of words

    """
    token = nltk.tokenize.RegexpTokenizer(r"\w+")
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    line = line.lower().split(" ")
    #fix contractions e.g. you're becomes you are
    line = [contractions.fix(word) for word in line]
    line=" ".join(line).lower()
    line = token.tokenize(line)
    #lemmatize
    line = [lemmatizer.lemmatize(word) for word in line]
    # remove whitespace
    line = [word.strip() for word in line]
    # Remove punctuations and numbers
    line = [re.sub('[^a-zA-Z]', ' ', word) for word in line]
    # Single character removal
    line = [re.sub(r"\s+[a-zA-Z]\s+", ' ', word) for word in line]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in line if word.isalpha()]
    
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
    if denom != 0:
        return num/denom
    else:
        return -1

def logreg(s,tokenizer, Xtrain, Xtest, Train_y, Test_y):
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
    Confusion matrix: numpy array
        output of logistic regression showing all predicted outcomes vs. human coded outcomes

    """
    set_random_seed(seed = s)
    Log = linear_model.LogisticRegression(random_state = s, max_iter = 10000)
    Log.fit(Xtrain,Train_y)
    # predict the labels on validation dataset
    predictions_Log = Log.predict(Xtest)
    words = []
    coefs = []
    #if s == 100:
    if False:
        #Log.coef_[2][i] for principles, Log.coef_[0][i] for limitations
        coefs_dict = {i: np.exp(Log.coef_[0][i]) for i in range(len(Log.coef_[0]))}
        sorted_keys = sorted(coefs_dict, key=coefs_dict.get) 
        words_dict = dict((v,k) for k,v in tokenizer.word_index.items())
        #for w in sorted_keys:
        #    sorted_dict[w] = coefs_dict[w]
        #print(sorted_dict)
        #print("\nNegative:")
        for num in sorted_keys[:20]:
            words.append(words_dict[num])
            coefs.append(coefs_dict[num])
        #print("\nPositive:")
        for num in sorted_keys[-20:]:
            words.append(words_dict[num])
            coefs.append(coefs_dict[num])
        print(words)
        print(coefs)
        #print(Log.predict_proba(Xtest))
        
    return confusion_matrix(Test_y, predictions_Log)
def get_predicts_binary(s, train_x,train_y, test_x):
    """
    Run the logistic regression for many trials, store and return all relevant data (e.g. computer est, human est, rate of false negatives and rate of false positives)
    
    For binary classification tasks. 
    
    Parameters
    ----------
    s : integer
        s is an integer specifying random seed.
    train_x : numpy array
        X (encoded responses) of training set
    train_y : numpy array
        y (classification) of training set
    test_x : numpy array
        data to be coded

    Returns
    -------
    predictions_Log : numpy array
        the model's classification of the new data set

    """
    tok = Tokenizer(lower = False)
    #tok.fit_on_texts(Train_X)
    tok.fit_on_texts(train_x)
    #Tr_X = tok.texts_to_matrix(Train_X, mode="binary")
    Tr_X = tok.texts_to_matrix(train_x, mode="binary")
    Te_X = tok.texts_to_matrix(test_x, mode="binary")
    #run the logistic regression
    set_random_seed(seed = s)
    Log = linear_model.LogisticRegression(random_state = s, max_iter = 10000)
    Log.fit(Tr_X,train_y)
    # predict the labels on validation dataset
    predictions_Log = Log.predict(Te_X)
    
    words = []
    coefs = []
    #Log.coef_[2][i] for principles, Log.coef_[0][i] for limitations
    coefs_dict = {i: np.exp(Log.coef_[0][i]) for i in range(len(Log.coef_[0]))}
    sorted_keys = sorted(coefs_dict, key=coefs_dict.get) 
    words_dict = dict((v,k) for k,v in tok.word_index.items())
    #for w in sorted_keys:
    #    sorted_dict[w] = coefs_dict[w]
    #print(sorted_dict)
    #print("\nNegative:")
    for num in sorted_keys[:10]:
        words.append(words_dict[num])
        coefs.append(coefs_dict[num])
    #print("\nPositive:")
    for num in sorted_keys[-20:]:
        words.append(words_dict[num])
        coefs.append(coefs_dict[num])
    print(words)
    print(coefs)
    return predictions_Log
    
    
def get_stats_est_fp_fn_binary(train_x, train_y, test_x, test_y, n, num_samples):
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
    n : int
        for each test, a sample of size n is pulled from the full test set

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
    #pull 100 samples from the full test set
    for s in range(num_samples):
        set_random_seed(seed = s)
        tok = Tokenizer(lower = False)
        #tok.fit_on_texts(Train_X)
        tok.fit_on_texts(train_x)
        #Tr_X = tok.texts_to_matrix(Train_X, mode="binary")
        Tr_X = tok.texts_to_matrix(train_x, mode="binary")
        #make test set, must have at least 2 examples of the code
        enough_pos_in_test = False
        i = 0
        if False:
            while not enough_pos_in_test:
                set_random_seed(seed = s + i*100)
                _, Te_X, _, Te_y = model_selection.train_test_split(test_x,test_y,test_size=n)
                if sum(Te_y)/len(Te_y) == 0.0 or sum(Te_y)/len(Te_y) == 1.0:
                    i +=1
                    #print(i)
                else:
                    enough_pos_in_test = True
                if i > 11:
                    print(i)
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _, Te_X, _, Te_y = model_selection.train_test_split(test_x,test_y,test_size=n)

        Te_X = tok.texts_to_matrix(Te_X, mode="binary")
        #run the logistic regression, store confusion matrix as mat
        mat = logreg(s,tok, Tr_X, Te_X, train_y, Te_y)
        if len(mat[0])<2:
            print("mat[0]<2")
            if sum(Te_y)== 0.0:
                mat = np.array([[n,0],[0,0]])
            else:
                mat = np.array([[0,0],[0,n]])

        #extract useful measures from mat
        est.append(mat[1][1] + mat[0][1])
        if sum(Te_y)/len(Te_y) == 0.0 or sum(Te_y)/len(Te_y) == 1.0:
            human_est.append(sum(Te_y))
        else:
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[1])
        fp.append(mat[0][1])
        fn.append(mat[1][0])
        kappa.append(cohens_kappa(mat[1][1], mat[1][0], mat[0][1], mat[0][0]))
        
    #add all data to dataframes and return
    df = pd.DataFrame({
        "est" : est,
        "fp" : fp,
        "fn" : fn,
        "kappa" : kappa,
        "human_est": human_est
    })
    if sum(Te_y)/len(Te_y) == 0.0 or sum(Te_y)/len(Te_y) == 1.0:
        df_human = pd.DataFrame({
            "train_y": pd.DataFrame(train_y).value_counts(sort = False),
            "test_y": sum(test_y)})
    else:
        df_human = pd.DataFrame({
            "train_y": pd.DataFrame(train_y).value_counts(sort = False),
            "test_y": pd.DataFrame(test_y).value_counts(sort = False)})
    
    return df, df_human

def get_stats_est_fp_fn_3code(train_x, train_y, test_x, test_y, val, n):
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
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data (trustworthy))
    n : int
        for each test, a sample of size n is pulled from the full test set
    
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
    #pull 100 samples from the full test set
    for s in range(100):
        set_random_seed(seed = s)
        tok = Tokenizer(lower = False)
        #tok.fit_on_texts(Train_X)
        tok.fit_on_texts(train_x)
        #Tr_X = tok.texts_to_matrix(Train_X, mode="binary")
        Tr_X = tok.texts_to_matrix(train_x, mode="binary")
        #make test set, must have at least 2 examples of the code
        enough_pos_in_test = False
        i = 0
        while not enough_pos_in_test:
            set_random_seed(seed = s + i*100)
            _, Te_X, _, Te_y = model_selection.train_test_split(test_x,test_y,test_size=n)
            vals = ["L","O","P"]
            if sum(1 for j in Te_y if j == vals[0])<2 or sum(1 for j in Te_y if j == vals[1])<2 or sum(1 for j in Te_y if j == vals[2])<2:
                i += 1
                print(i)
            else:
                enough_pos_in_test = True
                
            if i > 21:
                return pd.DataFrame(), pd.DataFrame()

        Te_X = tok.texts_to_matrix(Te_X, mode="binary")
        mat = logreg(s,tok, Tr_X, Te_X, train_y, Te_y)
        #list of 3 codes in alphabetical order
        if val == vals[0]:
            est.append(mat[0][0] + mat[1][0] + mat[2][0])
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[0])
            fp.append(mat[1][0] + mat[2][0])
            fn.append(mat[0][1] + mat[0][2])
            kappa.append(cohens_kappa(mat[0][0], fp[-1], fn[-1], mat[1][1]+ mat[2][2]))
        elif val == vals[1]:
            est.append(mat[0][1] + mat[1][1] + mat[2][1])
            human_est.append(pd.DataFrame(Te_y).value_counts(sort = False)[1])
            fp.append(mat[0][1] + mat[2][1])
            fn.append(mat[1][0] + mat[1][2])
            kappa.append(cohens_kappa(mat[1][1], fp[-1], fn[-1], mat[0][0]+ mat[2][2]))
        elif val == vals[2]:
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
def trustworthy_process( s, code, split_test):
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

    N_each, int

    tuple of X for each data subset

    tuple of y for each data subset
 
    """
    #read in the data
    df_full = pd.read_csv(r'trustworthy_dat.csv')
    #create equal sized, shuffled pre and post data frames. these are fixed and do not depend on seed.
    if split_test == "F22":
        df  = df_full[df_full["Semester"] != "F2022"]
        F22 = df_full[df_full["Semester"] == "F2022"]
    else:
        df = df_full
    POST = df[df["PRE/POST"] == "POST"]
    PRE = df[df["PRE/POST"] == "PRE"]
    
    set_random_seed(seed = s)
    POST = POST.sample(frac=1).reset_index(drop=True)
    PRE = PRE.sample(frac=1).reset_index(drop=True)
    #print(len(PRE))
    #print(len(POST))
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
    X_F22 = []
#    if split_test == "F22":
#        for response in F22["Trustworthy Response"].tolist():
#            X_F22.append((preprocess_text(response), F22["ResponseID"])) 
#        return (PRE, POST), N_each, (X_PRE, X_POST), (y_PRE, y_POST), X_F22
    if split_test == "F22":
        for index, row in F22.iterrows():
            X_F22.append((preprocess_text(row["Trustworthy Response"]), row["ResponseID"]))
        return (PRE, POST), N_each, (X_PRE, X_POST), (y_PRE, y_POST), X_F22
    return (PRE, POST), N_each, (X_PRE, X_POST), (y_PRE, y_POST)

def sources_process( s, code, val , systematic = "all"):
    """
    Pre-process sources of uncertainty data, break into groups of systematic variable.
    
    Data subsets as subsets defined by systematic variable of interest (exp): PM, BM, SG, SS. 

    Parameters
    ----------
    s : integer
        s is an integer specifying random seed.
    code : string
        the category of response in human coded data.
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data (trustworthy))
    systematic: string
        denotes how data is split based on a variable, "all" for no split/all data

    Returns
    -------
    tuple of full data frame for each data subset (subsets defined by systematic variable of interest), tuple

    N_each, int

    tuple of X for each data subset

    tuple of y for each data subset

    """
    #read in the data
    df = pd.read_csv(r'PLO_dat.csv')
    df = df[df["Q"].notnull()]
    df = df[df["Q"].str.len()>1]
    
    if systematic == "upper":
        df = df[df["intro/upper"]== "upper"]
    elif systematic == "intro":
        df = df[df["intro/upper"]== "intro"]
        
    if systematic == "male":
        df = df[df["Gender"]== "Male"]
    elif systematic == "gender-min":
        df = pd.concat([df[df["Gender"]== "Female"], df[df["Gender"]== "Non-binary"]])
        
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
    """
    #Pull out only the first 239 from each experiment (because min response number 239 for SS)
    PMs = PMs.iloc[:239]
    BMs = BMs.iloc[:239]
    SGs = SGs.iloc[:239]
    SSs = SSs.iloc[:239]
    #Keep the 468-239 additional PM responses because those will be useful
    PM_add = PMs.iloc[239:]
    BM_add = BMs.iloc[239:]
    """

    N_each = len(PMs)
    if systematic == "all":
        assert(N_each*4 - 200 > 700)
    
    #pre process responses (X) and match to code value (y) 
    X_PMs = []
    for response in PMs["Q"].tolist():
        X_PMs.append(preprocess_text(response))
    y_PMs = np.array(PMs[code].tolist())
    if val == 1:
        is_L = [1 if row == "L" else 0 for row in PMs[code]]
        y_PMs = np.array(is_L)
    else:
        y_PMs = np.array(PMs[code].tolist())
    
    if systematic != "intro" and systematic != "upper":
        X_BMs = []
        for response in BMs["Q"].tolist():
            X_BMs.append(preprocess_text(response))
        if val == 1:
            is_L = [1 if row == "L" else 0 for row in BMs[code]]
            y_BMs = np.array(is_L)
        else:
            y_BMs = np.array(BMs[code].tolist())
        
        X_SGs = []
        for response in SGs["Q"].tolist():
            X_SGs.append(preprocess_text(response))
        if val == 1:
            is_L = [1 if row == "L" else 0 for row in SGs[code]]
            y_SGs = np.array(is_L)
        else:
            y_SGs = np.array(SGs[code].tolist())
        
        X_SSs = []
        for response in SSs["Q"].tolist():
            X_SSs.append(preprocess_text(response))
        if val == 1:
            is_L = [1 if row == "L" else 0 for row in SSs[code]]
            y_SSs = np.array(is_L)
        else:
            y_SSs = np.array(SSs[code].tolist())
        
        
        return (PMs, BMs, SGs, SSs), N_each, (X_PMs, X_BMs, X_SGs, X_SSs), (y_PMs, y_BMs, y_SGs, y_SSs)
    elif systematic == "upper" or systematic == "intro":
        return (PMs,), N_each, (X_PMs,), (y_PMs,)

def get_data(train_dec, test_dec, code, val, s, n_full, n, train_size, split_test = "all", opt_trustworthy = False, num_samples = 100):
    """
    code is a string, the category of response in human coded data
    N_each is the number of responses after cleaning 

    Parameters
    ----------
    train_dec : float
        Number between 0 and 1 specifying fraction of responses with the code to put in training set.
    test_dec : float
        Number between 0 and 1 specifying fraction of responses with the code to put in test set.
    code : string
        the category of response in human coded data.
    val : string or int
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data (trustworthy))
    s : integer
        s is an integer specifying random seed.
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
    df : Pandas dataframe
        Dataframe of all trials (default 100) for given parameters. 
        
        All data at this stage has been run through the test_bank. 

    """

    if opt_trustworthy and split_test == "F22":
        df_s, N_each, X_s, y_s, X_F22 =  trustworthy_process(s, code, split_test)
        #train_size = 600
    elif opt_trustworthy:
        df_s, N_each, X_s, y_s =  trustworthy_process(s, code, split_test)
        #train_size = 600
    elif split_test == "upper" or split_test == "intro":
        df_s, N_PM, X_s, y_s = sources_process(s, code, val, systematic = "intro")
        df_s_sys, N_PM_sys, X_s_sys, y_s_sys = sources_process(s, code, val, systematic = "upper")
        #train_size = 550
        print(N_PM)
        print(N_PM_sys)
    elif split_test == "male" or split_test == "gender-min":
        df_s, N_each, X_s, y_s = sources_process(s, code, val, systematic = "male")
        df_s_sys, N_each, X_s_sys, y_s_sys = sources_process(s, code, val, systematic = "gender-min")
        #train_size = 600
    else:
        df_s, N_each, X_s, y_s = sources_process(s, code, val, systematic = split_test)
        #train_size = 600
    
    #number of pos and neg in test and training sets
    N_pos_test = math.floor(n_full*test_dec)
    N_neg_test = n_full - N_pos_test
    N_pos_train = math.floor(train_size*train_dec)
    N_neg_train = train_size - N_pos_train
    
    print("\n")
    print("N_pos_test " + str(N_pos_test))
    print("N_neg_test " + str(N_neg_test))
    print("N_pos_train " + str(N_pos_train))
    print("N_neg_train " + str(N_neg_train))
                
    if split_test == "pre" or split_test == "postpre":
        #create the data structures
        X_pre_pos = []
        X_pre_neg = []
        y_pre_pos = []
        y_pre_neg = []
        
        X_post_pos = []
        X_post_neg = []
        y_post_pos = []
        y_post_neg = []
        
        X_pre = X_s[0]
        X_post = X_s[1]
        
        y_pre = y_s[0]
        y_post = y_s[1]
        
        for idx, condition_met in enumerate(y_pre == val):
            if condition_met:
                X_pre_pos.append(X_pre[idx])
                y_pre_pos.append(y_pre[idx])
            elif not condition_met:
                X_pre_neg.append(X_pre[idx])
                y_pre_neg.append(y_pre[idx])
        for idx, condition_met in enumerate(y_post == val):
            if condition_met:
                X_post_pos.append(X_post[idx])
                y_post_pos.append(y_post[idx])
            elif not condition_met:
                X_post_neg.append(X_post[idx])
                y_post_neg.append(y_post[idx])
                
        #check validity of data structures
        assert(sum(1 for i in y_pre_pos if i == val)/len(y_pre_pos) == 1.0)
        assert(sum(1 for i in y_pre_neg if i == val)/len(y_pre_neg) == 0.0)
        assert(sum(1 for i in y_post_pos if i == val)/len(y_post_pos) == 1.0)
        assert(sum(1 for i in y_post_neg if i == val)/len(y_post_neg) == 0.0)
        
        random.shuffle(X_pre_pos)
        random.shuffle(X_pre_neg)
        random.shuffle(X_post_pos)
        random.shuffle(X_post_neg)
        
        
        print("length X_pre_pos " + str(len(X_pre_pos)))
        print("length X_pre_neg " + str(len(X_pre_neg)))
        print("length X_post_pos " + str(len(X_post_pos)))
        print("length X_post_neg " + str(len(X_post_neg)))
        
        #create training and test sets
        percent_pre = 0.2
        
        N_pos_train_pre = math.ceil(N_pos_train*percent_pre)
        N_pos_train_post = math.floor(N_pos_train*(1-percent_pre))
        N_neg_train_pre = math.ceil(N_neg_train*percent_pre)
        N_neg_train_post = math.floor(N_neg_train*(1-percent_pre))
        
        N_pos_test_pre = math.floor(N_pos_test*percent_pre)
        N_pos_test_post = math.ceil(N_pos_test*(1-percent_pre))
        N_neg_test_pre = math.floor(N_neg_test*percent_pre)
        N_neg_test_post = math.ceil(N_neg_test*(1-percent_pre))
        
        if len(X_pre_pos) < N_pos_train_pre + N_pos_test_pre or len(X_post_pos) < N_pos_train_post + N_pos_test_post  or len(X_pre_neg)  < N_neg_train_pre + N_neg_test_pre or len(X_post_neg) < N_neg_train_post + N_neg_test_post:
            return pd.DataFrame()
        
        #create X and y for fixed training set
        Train_X = X_pre_pos[:N_pos_train_pre] + X_pre_neg[:N_neg_train_pre] + X_post_pos[:N_pos_train_post] + X_post_neg[:N_neg_train_post]
        Train_y = np.concatenate((y_pre_pos[:N_pos_train_pre], y_pre_neg[:N_neg_train_pre], y_post_pos[:N_pos_train_post], y_post_neg[:N_neg_train_post]))
        
        if split_test == "pre":
            if len(X_pre_pos) < N_pos_train_pre + N_pos_test or len(X_pre_neg) < N_neg_train_pre + N_neg_test:
                return pd.DataFrame()
            full_test_X = X_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test] + X_pre_neg[N_neg_train_pre:N_neg_train_pre + N_neg_test]
            full_test_y = np.concatenate((y_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test], y_pre_neg[N_neg_train_pre:N_neg_train_pre + N_neg_test]))
        elif split_test == "postpre":
            if len(X_pre_pos) < N_pos_train_pre + N_pos_test_pre or len(X_post_pos) < N_pos_train_post + N_pos_test_post  or len(X_pre_neg)  < N_neg_train_pre + N_neg_test_pre or len(X_post_neg) < N_neg_train_post + N_neg_test_post:
                return pd.DataFrame()
            full_test_X = X_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test_pre] + X_pre_neg[N_neg_train_pre: N_neg_train_pre + N_neg_test_pre] +  X_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test_post] + X_post_neg[N_neg_train_post:N_neg_train_post  + N_neg_test_post]
            full_test_y = np.concatenate((y_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test_pre], y_pre_neg[N_neg_train_pre: N_neg_train_pre + N_neg_test_pre], y_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test_post], y_post_neg[N_neg_train_post: N_neg_train_post + N_neg_test_post]))
    
    elif split_test == "all" or split_test == "F22":
    #if not testing population systematics
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
        
        if split_test == "F22":
            full_test_X = []
            corresponding_responseID = []
            for row in X_F22:
                full_test_X.append(row[0])
                corresponding_responseID.append(row[1])
            responseID = X_F22
            codes = get_predicts_binary(s, Train_X, Train_y, full_test_X)
            #"Response": full_test_X
            return pd.DataFrame({"Response": full_test_X , code: codes, "ResponseID": corresponding_responseID})
           
        full_test_X = X_pos[N_pos_train:N_pos_train + N_pos_test] + X_neg[N_neg_train: N_neg_train + N_neg_test]
        full_test_y = np.concatenate((y_pos[N_pos_train:N_pos_train + N_pos_test], y_neg[N_neg_train: N_neg_train + N_neg_test]))
         
    
    #now that train and test sets are made, print their size
    print("train X " + str(len(Train_X)))
    print("train y " + str(len(Train_y)))
    print("test X " + str(len(full_test_X)))
    print("test y " + str(len(full_test_y)))
    
    
    #Run logreg on 100 test set samples (equal test set)
    if val == 1:
        df, df_human = get_stats_est_fp_fn_binary(Train_X, Train_y, full_test_X, full_test_y, n, num_samples)
    else:
        df, df_human = get_stats_est_fp_fn_3code(Train_X, Train_y, full_test_X, full_test_y, val, n)
    if df.empty:
        return pd.DataFrame()
    if sum(1 for i in Train_y if i == val)<2:
        warnings.warn("Less than 2 examples of code in Training set")
    
    test_bank.tests_on_inputs(df_s, Train_X, Train_y, full_test_X, full_test_y, val, opt_trustworthy )
    #might be able to not output the human dfs because of the tests, write down where that info can be gathered elsewhere
    test_bank.tests_on_outputs(df["est"], df["human_est"],df["fp"], df["fn"], df_human, test_dec, train_dec, n_full, n)
    print(df_human)
    return df


"""
    else:
            
            percent_pre = 0.5
            
            N_pos_train_pre = math.ceil(N_pos_train*percent_pre)
            N_pos_train_post = math.floor(N_pos_train*(1-percent_pre))
            N_neg_train_pre = math.ceil(N_neg_train*percent_pre)
            N_neg_train_post = math.floor(N_neg_train*(1-percent_pre))
            
            N_pos_test_pre = math.floor(N_pos_test*percent_pre)
            N_pos_test_post = math.ceil(N_pos_test*(1-percent_pre))
            N_neg_test_pre = math.floor(N_neg_test*percent_pre)
            N_neg_test_post = math.ceil(N_neg_test*(1-percent_pre))
            
            if len(X_pre_pos) < N_pos_train_pre + N_pos_test_pre or len(X_post_pos) < N_pos_train_post + N_pos_test_post  or len(X_pre_neg)  < N_neg_train_pre + N_neg_test_pre or len(X_post_neg) < N_neg_train_post + N_neg_test_post:
                return pd.DataFrame()
            
            #create X and y for fixed training set
            Train_X = X_pre_pos[:N_pos_train_pre] + X_pre_neg[:N_neg_train_pre] + X_post_pos[:N_pos_train_post] + X_post_neg[:N_neg_train_post]
            Train_y = np.concatenate((y_pre_pos[:N_pos_train_pre], y_pre_neg[:N_neg_train_pre], y_post_pos[:N_pos_train_post], y_post_neg[:N_neg_train_post]))
            
            if split_test == "F22":
                full_test_X = X_F22
                codes = get_predicts_binary(s, Train_X, Train_y, full_test_X)
                #"Response": full_test_X
                return pd.DataFrame({"Response": full_test_X ,"Uncertainty": codes})
            elif split_test == "post":
                if len(X_post_pos) < N_pos_train_post + N_pos_test or len(X_post_neg) < N_neg_train_post + N_neg_test:
                    return pd.DataFrame()
                full_test_X = X_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test] + X_post_neg[N_neg_train_post:N_neg_train_post + N_neg_test]
                full_test_y = np.concatenate((y_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test], y_post_neg[N_neg_train_post:N_neg_train_post + N_neg_test]))
            else:
                full_test_X = X_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test_pre] + X_pre_neg[N_neg_train_pre: N_neg_train_pre + N_neg_test_pre] +  X_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test_post] + X_post_neg[N_neg_train_post:N_neg_train_post  + N_neg_test_post]
                full_test_y = np.concatenate((y_pre_pos[N_pos_train_pre:N_pos_train_pre + N_pos_test_pre], y_pre_neg[N_neg_train_pre: N_neg_train_pre + N_neg_test_pre], y_post_pos[N_pos_train_post:N_pos_train_post + N_pos_test_post], y_post_neg[N_neg_train_post: N_neg_train_post + N_neg_test_post]))
            
 """   