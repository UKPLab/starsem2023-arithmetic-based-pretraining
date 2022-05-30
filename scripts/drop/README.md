# DROP Preprocessing

This directory contains the scripts necessary to preprocess the DROP dataset to be usable with our code. DROP is provided in .json files and can be downloaded [here](https://allenai.org/data/drop). This script convert the paragraphs and questions into a linearized representation and assigns them with the corresponding answer:

1. preprocess_drop.py

If you want to create data for the INP task, you have to execute the following script:

2. preprocess_wikibio_inp.py