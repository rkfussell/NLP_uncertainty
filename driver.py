# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:44:43 2023

@author: rkf33
"""

#import test_bank
#test_bank.tests_on_data()


#Create data files
import pull_data_code_seed as pull_data

pull_data.get_data_code_seed("CON", 1, 10, "Networks_test", "Cornell",  train_sizes = [700], train_decs = [0.5], test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], n_fulls = [100], ns = [50], num_samples = 100)
pull_data.get_data_code_seed("CON", 1, 10, "Networks_test", "UTAustin",  train_sizes = [700], train_decs = [0.5], test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], n_fulls = [100], ns = [50], num_samples = 100)
pull_data.get_data_code_seed("CON", 1, 10, "Networks_test", "NCState",  train_sizes = [700], train_decs = [0.5], test_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], n_fulls = [100], ns = [50], num_samples = 100)

