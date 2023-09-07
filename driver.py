# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:44:43 2023

@author: rkf33
"""

#import test_bank
#test_bank.tests_on_data()


#Create data files
import pull_data_code_seed as pull_data

pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_vary_train_size_", train_sizes = [100,150,200,250,300,350,400,450,500,550,600], train_decs = [0.5], test_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("PLO", 1, 1, "figures_vary_train_size_", train_sizes = [100,150,200,250,300,350,400,450,500,550,600], train_decs = [0.5], test_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = False)
