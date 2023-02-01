# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:44:43 2023

@author: rkf33
"""

#import test_bank
#test_bank.tests_on_data()


#Create data files
import pull_data_code_seed as pull_data

pull_data.get_data_code_seed("PLO", "L", 1, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", "P", 1, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", "L", 2, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", "P", 2, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", "L", 3, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", "P", 3, opt_trustworthy = False)

pull_data.get_data_code_seed("Expected", 1, 1, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 1, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 1, opt_trustworthy = True)
pull_data.get_data_code_seed("Statistics", 1, 1, opt_trustworthy = True)

pull_data.get_data_code_seed("Expected",1, 2, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 2, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 2, opt_trustworthy = True)
pull_data.get_data_code_seed("Statistics",1, 2, opt_trustworthy = True)

pull_data.get_data_code_seed("Expected",1, 3, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results",1,3, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 3, opt_trustworthy = True)
pull_data.get_data_code_seed("Statistics",1, 3, opt_trustworthy = True)
