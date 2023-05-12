# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:57:26 2023

@author: rkf33
"""
import pandas as pd
"""
#Add all intro vs upper dataframes to a single dataframe with column seed
df_full = pd.DataFrame()
for s in range(100):
    df1 = pd.read_csv("intro_vs_upper_upper_codePLOval1seed" + str(s) + "_df.csv")
    df1["seed"] = s
    df1["introupper"] = "upper"
    df2 = pd.read_csv("intro_vs_upper_intro_codePLOval1seed" + str(s) + "_df.csv")
    df2["seed"] = s
    df2["introupper"] = "intro"
    df_full = pd.concat([df_full, df1, df2])
df_full.to_csv("intro_vs_upper_full_df.csv")

#Add all intro vs upper dataframes to a single dataframe with column seed
df_full = pd.DataFrame()
for s in range(100):
    df1 = pd.read_csv("male_vs_gender-min_male_codePLOval1seed" + str(s) + "_df.csv")
    df1["seed"] = s
    df1["gender"] = "male"
    df2 = pd.read_csv("male_vs_gender-min_gender-min_codePLOval1seed" + str(s) + "_df.csv")
    df2["seed"] = s
    df2["gender"] = "gender-min"
    df_full = pd.concat([df_full, df1, df2])
df_full.to_csv("male_vs_gender-min_full_df.csv")
"""
df_full = pd.DataFrame()
for s in range(100):
    df1 = pd.read_csv("post_vs_pre_pre_codeConsistent Resultsval1seed" + str(s) + "_df.csv")
    df1["seed"] = s
    df1["prepost"] = "pre"
    df2 = pd.read_csv("post_vs_pre_post_codeConsistent Resultsval1seed" + str(s) + "_df.csv")
    df2["seed"] = s
    df2["prepost"] = "post"
    df_full = pd.concat([df_full, df1, df2])
df_full.to_csv("pre_vs_post_full_df.csv")