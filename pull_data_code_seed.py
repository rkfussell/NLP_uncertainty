# -*- coding: utf-8 -*-
"""

@author: - Rebeckah Fussell
"""

import os

print(os.getcwd())

import get_systematics_data as gsd

def get_data_code_seed(s = 1, code = "Expected"):
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
    
#get_data_code_seed(s = 1, code = "Expected")
get_data_code_seed(s = 1, code = "Consistent Results")
get_data_code_seed(s = 1, code = "Good Methods")
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