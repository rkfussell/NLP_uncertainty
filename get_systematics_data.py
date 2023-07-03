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

     
def explanations_process( s, code):
    """
    Pre-process explanations data

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
    df = pd.read_csv('Networks_explanations_dat.csv')
    #create equal sized, shuffled pre and post data frames. these are fixed and do not depend on seed.
   

    CORNELLTRAIN = df[df["Group"] == "Train:Cornell"]
    CORNELL = df[df["Group"] == "Cornell"]
    UTAUSTIN = df[df["Group"] == "UT"]
    NCSTATE = df[df["Group"] == "NCState"]
    
    
    
    set_random_seed(seed = s)
    
    CORNELLTRAIN = CORNELLTRAIN.sample(frac=1).reset_index(drop=True)
    CORNELL = CORNELL.sample(frac=1).reset_index(drop=True)
    UTAUSTIN = UTAUSTIN.sample(frac=1).reset_index(drop=True)
    NCSTATE = NCSTATE.sample(frac=1).reset_index(drop=True)
    
    
    #pre process responses (X) and match to code value (y) 
    X_CORNELLTRAIN = []
    for response in CORNELLTRAIN["explanation"].tolist():
        X_CORNELLTRAIN.append(preprocess_text(response))
    y_CORNELLTRAIN = np.array(CORNELLTRAIN[code].tolist())
    
    X_CORNELL = []
    for response in CORNELL["explanation"].tolist():
        X_CORNELL.append(preprocess_text(response))
    y_CORNELL = np.array(CORNELL[code].tolist())
    
    X_UTAUSTIN = []
    for response in UTAUSTIN["explanation"].tolist():
        X_UTAUSTIN.append(preprocess_text(response))
    y_UTAUSTIN = np.array(UTAUSTIN[code].tolist())
    
    X_NCSTATE = []
    for response in NCSTATE["explanation"].tolist():
        X_NCSTATE.append(preprocess_text(response))
    y_NCSTATE = np.array(NCSTATE[code].tolist())

    return (CORNELLTRAIN, CORNELL, UTAUSTIN, NCSTATE), (X_CORNELLTRAIN, X_CORNELL, X_UTAUSTIN, X_NCSTATE), (y_CORNELLTRAIN, y_CORNELL, y_UTAUSTIN, y_NCSTATE)

def get_data(train_dec, test_dec, code, val, s, n_full, n, train_size, test_institution, num_samples = 100):
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
        estimate frequency in dataset of this value of code (e.g. "L" or 1 for binary data)
    s : integer
        s is an integer specifying random seed.
    n_full: int
        number of responses in the full test set 
    n : int
        for each test, a sample of size n is pulled from the full test set
    train_size: int
        number of responses in the training set
    Returns
    -------
    df : Pandas dataframe
        Dataframe of all trials (default 100) for given parameters. 
        
        All data at this stage has been run through the test_bank. 

    """
    
   
    df_s, X_s, y_s = explanations_process(s, code)
    
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
                
    (df_train, df_CU, df_UT, df_NCS)  = df_s
    (X_train, X_CU, X_UT, X_NCS) = X_s
    (y_train, y_CU, y_UT, y_NCS) = y_s    


    X_pos_train = []
    X_neg_train = []
    y_pos_train = []
    y_neg_train = []

    
    for idx, condition_met in enumerate(y_train == val):
        if condition_met:
            X_pos_train.append(X_train[idx])
            y_pos_train.append(y_train[idx])
        elif not condition_met:
            X_neg_train.append(X_train[idx])
            y_neg_train.append(y_train[idx])   
    assert(sum(1 for i in y_pos_train if i == val)/len(y_pos_train) == 1.0)
    assert(sum(1 for i in y_neg_train if i == val)/len(y_neg_train) == 0.0)
    
    random.shuffle(X_pos_train)
    random.shuffle(X_neg_train)
    
    if len(X_pos_train) <  N_pos_train or len(X_neg_train) <  N_neg_train:
        return pd.DataFrame()
    
    X_pos = []
    X_neg = []
    y_pos = []
    y_neg = []
    
    if test_institution == "Cornell":
        df = df_CU
        X = X_CU
        y = y_CU
    if test_institution == "UTAustin":
        df = df_UT
        X = X_UT
        y = y_UT
    if test_institution == "NCState":
        df = df_NCS
        X = X_NCS
        y = y_NCS

    
    for idx, condition_met in enumerate(y == val):
        if condition_met:
            X_pos.append(X[idx])
            y_pos.append(y[idx])
        elif not condition_met:
            X_neg.append(X[idx])
            y_neg.append(y[idx])   
    assert(sum(1 for i in y_pos if i == val)/len(y_pos) == 1.0)
    assert(sum(1 for i in y_neg if i == val)/len(y_neg) == 0.0)
    
    
    random.shuffle(X_pos)
    random.shuffle(X_neg)
    
    print("length X_pos " + str(len(X_pos)))
    print("length X_neg " + str(len(X_neg)))
    
    if len(X_pos) <  N_pos_test or len(X_neg) <  N_neg_test:
        return pd.DataFrame()
    
    #create X and y for fixed training set
    Train_X = X_pos_train[:N_pos_train] + X_neg_train[:N_neg_train]
    Train_y = np.concatenate((y_pos_train[:N_pos_train],y_neg_train[:N_neg_train]))
    
    #create X and y for full test set
    full_test_X = X_pos[:N_pos_test] + X_neg[:N_neg_test]
    full_test_y = np.concatenate((y_pos[:N_pos_test],y_neg[:N_neg_test]))
    
    #now that train and test sets are made, print their size
    print("train X " + str(len(Train_X)))
    print("train y " + str(len(Train_y)))
    print("test X " + str(len(full_test_X)))
    print("test y " + str(len(full_test_y)))
    
    
    #Run logreg on 100 test set samples (equal test set)
    df, df_human = get_stats_est_fp_fn_binary(Train_X, Train_y, full_test_X, full_test_y, n, num_samples)


    if df.empty:
        return pd.DataFrame()
    if sum(1 for i in Train_y if i == val)<2:
        warnings.warn("Less than 2 examples of code in Training set")
    
    test_bank.tests_on_inputs(df_s, Train_X, Train_y, full_test_X, full_test_y, val)
    #might be able to not output the human dfs because of the tests, write down where that info can be gathered elsewhere
    test_bank.tests_on_outputs(df["est"], df["human_est"],df["fp"], df["fn"], df_human, test_dec, train_dec, n_full, n)
    print(df_human)
    return df


