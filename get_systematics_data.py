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
    if s==0:
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
            Te_X = tok.texts_to_matrix(Te_X, mode="binary")
            mat = logreg1(s,tok, Tr_X, Te_X, train_y, Te_y)
            if sum(Te_y)<2:
                n +=1
                print(n)
            else:
                enough_pos_in_test = True

        
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


def get_data_trust(s = 1, code = "Expected", train_rep = "equal"):
    #read in the data
    df = pd.read_excel (r'Trustworthy_Master_Spreadsheet_Summer_2022.xlsx')
    df["Trustworthy Response"] = df["Trustworthy Response"].str.replace(".","")
    print(df)
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
    #keep 820 from both PRE and POST because there are only 820 in PRE (same every time, fixed at seed 132)
    PRE = PRE[:820]
    POST = POST[:820]
    
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
    
    
    #decide what to do next based on train_rep specified by user
    if train_rep == "equal":
        #CREATE EQUAL REP TRAINING SET
        enough_pos_train = False
        Train_X = X_PRE[:400] + X_POST[:400]
        Train_y = np.concatenate((y_PRE[:400],y_POST[:400]))
        #create a test set that is sampled later
        full_test_X = X_PRE[400:500] + X_POST[400:500]
        full_test_y = np.concatenate((y_PRE[400:500], y_POST[400:500]))
        full_test_X_PRE = X_PRE[500:700]
        full_test_y_PRE = y_PRE[500:700]
    elif train_rep == "over":
        #CREATE OVER-REP PRE TRAINING SET
        Train_X = X_PRE[:600] + X_POST[:200]
        Train_y = np.concatenate((y_PRE[:600],y_POST[:200]))
        #create a test set that is sampled later
        full_test_X = X_PRE[600:700] + X_POST[200:300]
        full_test_y = np.concatenate((y_PRE[600:700],y_POST[200:300]))
        full_test_X_PRE = X_PRE[600:800]
        full_test_y_PRE = y_PRE[600:800]
    elif train_rep == "under":
        Train_X = X_PRE[:200] + X_POST[:600]
        Train_y = np.concatenate((y_PRE[:200],y_POST[:600]))
        #create a test set that is sampled later
        full_test_X = X_PRE[200:300] + X_POST[600:700]
        full_test_y = np.concatenate((y_PRE[200:300],y_POST[600:700]))
        full_test_X_PRE = X_PRE[600:800]
        full_test_y_PRE = y_PRE[600:800]
    #Run logreg on 100 test set samples (equal test set)
    df, df_human = get_stats_est_fp_fn_trust(Train_X, Train_y, full_test_X, full_test_y)
    #Run logreg on 100 test set samples (PRE-only test set)
    df_PRE, df_human_PRE = get_stats_est_fp_fn_trust(Train_X, Train_y, full_test_X_PRE, full_test_y_PRE)
    if sum(Train_y)<2:
        warnings.warn("Less than 2 examples of code in Training set")
    test_bank.tests_on_inputs(PRE, POST, Train_X, Train_y, full_test_X, full_test_y, full_test_X_PRE, full_test_y_PRE, opt_trustworthy = True)
    #might be able to not output the human dfs because of the tests, write down where that info can be gathered elsewhere
    test_bank.tests_on_outputs(df["est"], df["human_est"],df["fp"], df["fn"])
    return df, df_human, df_PRE, df_human_PRE
