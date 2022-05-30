# SciGen Preprocessing

This directory contains the scripts necessary to preprocess the SciGen dataset to be usable with our code. SciGen is provided in .json files and can be downloaded [here](https://github.com/UKPLab/SciGen/tree/main/dataset). These scripts (1) convert the tables from their json format into a linearized representation, and (2) assign the tables to their targets and store the results in corresponding .source and .target files.

The scripts should be executed in the following order:

1. preprocess_scigen.py

If you want to create data for the INP task, you have to execute the following script:

2. preprocess_scigen_inp.py