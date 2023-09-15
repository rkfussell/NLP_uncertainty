# NLP_uncertainty
Code is run from driver.py
driver.py calls function in pull_data_code_seed.py
pull_data_code_seed.py calls function in get_systematics_data.py (the bulk of the experiment happens in get_systematics_data.py)

test_bank.py is called by pull_data_code_seed.py, runs checks to make sure code does what it is supposed to
clean_trustworthy.py is called by pull_data_code_seed.py, is specific to our Trustworthy data source
