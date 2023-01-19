# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os
import pandas as pd

print(os.getcwd())

import get_systematics_data as gsd

def get_data_code_seed_OLD(s = 1, code = "Expected"):
    equal_df, equal_df_human, equal_df_PRE, equal_df_human_PRE = gsd.get_data_trust(s = s, code =  code,train_rep = "equal")
    over_df, over_df_human, over_df_PRE, over_df_human_PRE = gsd.get_data_trust(s = s, code =  code,train_rep = "over")
    under_df, under_df_human, under_df_PRE, under_df_human_PRE = gsd.get_data_trust(s = s, code =  code,train_rep = "under")
    
    equal_df.to_csv(code + str(s) + "_equal_df.csv")
    equal_df_human.to_csv(code + str(s) +"_equal_df_human.csv")
    equal_df_PRE.to_csv(code + str(s) +"_equal_df_PRE.csv")
    equal_df_human_PRE.to_csv(code + str(s) +"_equal_df_human_PRE.csv")
    
    over_df.to_csv(code + str(s) + "_over_df.csv")
    over_df_human.to_csv(code + str(s) +"_over_df_human.csv")
    over_df_PRE.to_csv(code + str(s) +"_over_df_PRE.csv")
    over_df_human_PRE.to_csv(code + str(s) +"_over_df_human_PRE.csv")
    
    under_df.to_csv(code + str(s) + "_under_df.csv")
    under_df_human.to_csv(code + str(s) +"_under_df_human.csv")
    under_df_PRE.to_csv(code + str(s) +"_under_df_PRE.csv")
    under_df_human_PRE.to_csv(code + str(s) +"_under_df_human_PRE.csv")
    
def get_data_train_test(train_dec,test_dec, s, code):
    df = gsd.get_data_trust(train_dec, test_dec, s = s, code =  code)
    df["train_prop"] = train_dec
    df["test_prop"] = test_dec
    return df

def get_data_code_seed(s = 1, code = "Expected"):
    train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    df = pd.DataFrame()
    for train_dec in train_decs:
        for test_dec in test_decs:
            df_new = get_data_train_test(train_dec, test_dec, s, code)
            if not df_new.empty:
                df = pd.concat([df, df_new])
    df.to_csv(code + str(s) + "code_perc_df.csv")


 
#get_data_code_seed(s = 1, code = "Expected")
#get_data_code_seed(s = 1, code = "Consistent Results")
#get_data_code_seed(s = 1, code = "Good Methods")
get_data_code_seed(s = 1, code = "Statistics")

get_data_code_seed(s = 2, code = "Expected")
get_data_code_seed(s = 2, code = "Consistent Results")
get_data_code_seed(s = 2, code = "Good Methods")
get_data_code_seed(s = 2, code = "Statistics")

get_data_code_seed(s = 3, code = "Expected")
get_data_code_seed(s = 3, code = "Consistent Results")
get_data_code_seed(s = 3, code = "Good Methods")
get_data_code_seed(s = 3, code = "Statistics")

#TEST 5: trials with same random seed are deterministically identical