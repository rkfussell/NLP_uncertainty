# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:33:03 2023

@author: rkf33
"""
import pandas as pd

df = pd.read_excel (r'Trustworthy_Master_Spreadsheet_Summer_2022.xlsx')
df["Trustworthy Response"] = df["Trustworthy Response"].str.replace(".","")
df = df[df["Trustworthy Response"].notnull()]
df = df[df["Trustworthy Response"].str.len()>1]
#remove duplicates from master spreadsheet
df = df[~df.duplicated(subset = "Trustworthy Response", keep = "first")]
df = df[df["ResponseID"]!=124]
df = df[df["ResponseID"]!=308]

df.to_csv("trustworthy_dat.csv")