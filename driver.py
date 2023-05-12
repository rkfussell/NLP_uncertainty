# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:44:43 2023

@author: rkf33
"""

#import test_bank
#test_bank.tests_on_data()


#Create data files
import pull_data_code_seed as pull_data

#pull_data.get_data_code_seed("Consistent Results", 1, 35, "figures_pop_systematics_", train_sizes = [500], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200], ns = [100], split_test = "pre", opt_trustworthy = True)
#pull_data.get_data_code_seed("Consistent Results", 1, 35, "figures_pop_systematics_", train_sizes = [500], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200], ns = [100], split_test = "postpre", opt_trustworthy = True)


#pull_data.get_data_code_seed("Consistent Results", 1, 2, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.5], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = True)

#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.5], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = True)

#for i in range(100):
#    pull_data.get_data_code_seed("Uncertainty", 1, i, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100], split_test = "post", opt_trustworthy = True)
#    pull_data.get_data_code_seed("Uncertainty", 1, i, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100], split_test = "postpre", opt_trustworthy = True)
#    #pull_data.get_data_code_seed("Uncertainty", 1, i, "code_F22", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)
pull_data.get_data_code_seed("Uncertainty", 1, 10, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100], split_test = "post", opt_trustworthy = True)

pull_data.get_data_code_seed("Uncertainty", 1, 4, "code_F22", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)



#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_vary_n_small", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [80,100,150,200,300], ns = [50,60,70,80,90,100], split_test = "all", opt_trustworthy = True)
#pull_data.get_data_code_seed("PLO", 1, 1, "figures_systematics_vary_n_small", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [80,100,150,200,300], ns = [50,60,70,80,90,100], split_test = "all", opt_trustworthy = False)





#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_fix_test_vary_train_", train_sizes = [600], train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = True)
#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_fix_test_vary_train_", train_sizes = [600], train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = True)


#pull_data.get_data_code_seed("Consistent Results", 1, 3, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = True)

#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_vary_train_size_", train_sizes = [100,150,200,250,300,350,400,450,500,550,600], train_decs = [0.5], test_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], n_fulls = [200], ns = [50,100], split_test = "all", opt_trustworthy = True)
#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_vary_n_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200], split_test = "all", opt_trustworthy = True)
#pull_data.get_data_code_seed("PLO", 1, 1, "figures_vary_train_size_", train_sizes = [100,150,200,250,300,350,400,450,500,550,600], train_decs = [0.5], test_decs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], n_fulls = [200], ns = [50,100], split_test = "all", opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", 1, 1, "figures_systematics_vary_n_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200], split_test = "all", opt_trustworthy = False)


#pull_data.get_data_code_seed("Uncertainty", 1, 3, split_test = "all", opt_trustworthy = True)

#pull_data.get_data_code_seed("Uncertainty", 1, 1, split_test = "post", opt_trustworthy = True)
#pull_data.get_data_code_seed("Uncertainty", 1, 2, split_test = "post", opt_trustworthy = True)
#pull_data.get_data_code_seed("Uncertainty", 1, 3, split_test = "post", opt_trustworthy = True)
"""
pull_data.get_data_code_seed("Consistent Results", 1, 1, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 2, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 3, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 1, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 2, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 3, split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("PLO", 1, 1, split_test = "all", opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", 1, 2, split_test = "all", opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", 1, 3, split_test = "all", opt_trustworthy = False)


pull_data.get_data_code_seed("PLO", 1, 1, upper_lower_split_test = "all", opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", "P", 1, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", 1, 2, upper_lower_split_test = "all", opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", "P", 2, opt_trustworthy = False)
pull_data.get_data_code_seed("PLO", 1, 3, upper_lower_split_test = "all", opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", "P", 3, opt_trustworthy = False)

#pull_data.get_data_code_seed("PLO", 1, 4, opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", 1, 5, opt_trustworthy = False)
#pull_data.get_data_code_seed("PLO", 1, 6, opt_trustworthy = False)

#pull_data.get_data_code_seed("Expected", 1, 1, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 1, upper_lower_split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 2, upper_lower_split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 3, upper_lower_split_test = "all", opt_trustworthy = True)

pull_data.get_data_code_seed("Good Methods", 1, 1, upper_lower_split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 2, upper_lower_split_test = "all", opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods", 1, 3, upper_lower_split_test = "all", opt_trustworthy = True)


#pull_data.get_data_code_seed("Consistent Results", 1, 1, split_test = "pre", opt_trustworthy = True)
#pull_data.get_data_code_seed("Consistent Results", 1, 1, split_test = "post", opt_trustworthy = True)

for i in range(100):
    #pull_data.get_data_code_seed("PLO", 1, i, split_test = "male", opt_trustworthy = False)
    #pull_data.get_data_code_seed("PLO", 1, i, split_test = "gender-min", opt_trustworthy = False)
    pull_data.get_data_code_seed("Consistent Results", 1, i, split_test = "pre", opt_trustworthy = True)
    pull_data.get_data_code_seed("Consistent Results", 1, i, split_test = "post", opt_trustworthy = True)
    #pull_data.get_data_code_seed("Good Methods", 1, i, split_test = "pre", opt_trustworthy = True)




pull_data.get_data_code_seed("Good Methods", 1, 1, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics", 1, 1, opt_trustworthy = True)


#pull_data.get_data_code_seed("Expected",1, 2, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 2, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 2, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics",1, 2, opt_trustworthy = True)

#pull_data.get_data_code_seed("Expected",1, 3, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results",1,3, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 3, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics",1, 3, opt_trustworthy = True)

#pull_data.get_data_code_seed("Expected",1, 4, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results",1,4, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 4, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics",1, 4, opt_trustworthy = True)

#pull_data.get_data_code_seed("Expected",1, 5, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results",1,5, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 5, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics",1, 5, opt_trustworthy = True)

#pull_data.get_data_code_seed("Expected",1, 6, opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results",1,6, opt_trustworthy = True)
pull_data.get_data_code_seed("Good Methods",1, 6, opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics",1, 6, opt_trustworthy = True)
"""