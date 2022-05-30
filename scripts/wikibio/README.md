# WikiBio Preprocessing

This directory contains the scripts necessary to preprocess the WikiBio dataset to be usable with our code. WikiBio is provided in .jsonl files and can be downloaded [here](https://rlebret.github.io/wikipedia-biography-dataset/). These scripts (1) convert the tables from their json format into a linearized representation, and (2) assign the tables to their captions and targets and store the results in corresponding .source and .target files.

The scripts should be executed in the following order:

1. preprocess_wikibio.py

If you want to create data for the INP task, you have to execute the following script:

2. preprocess_wikibio_inp.py