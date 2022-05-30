# InfoTabs Preprocessing

This directory contains the scripts necessary to preprocess the InfoTabs dataset to be usable with our code. InfoTabs is provided in .tsv files and with tables separated in json files. It can be downloaded [here](https://github.com/infotabs/infotabs). These scripts (1) convert the separated tables from their json format into a linearized representation, and (2) assign the tables to their hypothesis and targets and store the results in corresponding .source and .target files.

The scripts should be executed in the following order:

1. convert_infotabs_tables.py
2. preprocess_infotabs.py

If you want to create data for the INP task, you have to execute the following script:

3. preprocess_infotabs_inp.py