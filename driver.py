# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:44:43 2023

@author: rkf33
"""

#import test_bank
#test_bank.tests_on_data()


#Create data files
import pull_data_code_seed as pull_data

pull_data.get_data_code_seed("Uncertainty", 1, 4, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [50], split_test = "postpre", opt_trustworthy = True)
pull_data.get_data_code_seed("Uncertainty", 1, 4, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [50], split_test = "post", opt_trustworthy = True)


#pull_data.get_data_code_seed("PLO", 1, 1, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = False, num_samples = 100)
#pull_data.get_data_code_seed("PLO", 1, 2, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = False, num_samples = 100)
#pull_data.get_data_code_seed("PLO", 1, 3, "figures_systematics_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [50,100], split_test = "all", opt_trustworthy = False, num_samples = 100)


#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_vary_n_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [100], split_test = "all", opt_trustworthy = True)
#for i in range(100):
#    pull_data.get_data_code_seed("Uncertainty", 1, i, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100], split_test = "post", opt_trustworthy = True)
#    pull_data.get_data_code_seed("Uncertainty", 1, i, "Apply_Method_", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100], split_test = "postpre", opt_trustworthy = True)
#    #pull_data.get_data_code_seed("Uncertainty", 1, i, "code_F22", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)

#pull_data.get_data_code_seed("Uncertainty", 1, 4, "code_F22", train_sizes = [400], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)
#pull_data.get_data_code_seed("Expected", 1, 4, "code_F22", train_sizes = [350], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)
#pull_data.get_data_code_seed("Statistics", 1, 4, "code_F22", train_sizes = [220], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)
#pull_data.get_data_code_seed("Ethics", 1, 4, "code_F22", train_sizes = [70], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)
#pull_data.get_data_code_seed("Peer Review", 1, 4, "code_F22", train_sizes = [28], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [150], ns = [100],split_test = "F22", opt_trustworthy = True)

#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_fix_test_vary_train_", train_sizes = [400], train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "pre", opt_trustworthy = True)
#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_fix_test_vary_train_", train_sizes = [400], train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "postpre", opt_trustworthy = True)
pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_vary_n_", train_sizes = [600], train_decs = [0.5], test_decs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_fulls = [200,300], ns = [100], split_test = "all", opt_trustworthy = True, num_samples = 1000)



#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_systematics_vary_n_small", train_sizes = [600], train_decs = [0.5], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = True)
#pull_data.get_data_code_seed("PLO", 1, 1, "figures_systematics_vary_n_small", train_sizes = [600], train_decs = [0.5], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = False)

#pull_data.get_data_code_seed("Consistent Results", 1, 1, "figures_fix_test_vary_train_", train_sizes = [600], train_decs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_decs = [0.5], n_fulls = [200], ns = [100], split_test = "all", opt_trustworthy = True)
