'''
    preprocess_infotabs_inp.py -- This script uses the preprocessed and linearized tables and converts it into data for the INP task (restrict the hypothesis to those that are labeled as entailed and contain numbers) 
'''

import pandas as pd
import os
import re

# This is the tsv file for the data split (e.g. "infotabs_test_alpha3.tsv" for 
# the third test split)
TSV_FILE = '/path/to/infotabs/data/infotabs_test_alpha1.tsv'

# This is the directory that contains the tables in linearized representation
TABLE_DIR = '/path/to/tables/preprocessed'

# This is the name of the output file (e.g. "test3" -> "test3.source", "test3.
# target")
OUTPUT_FILENAME = 'output_filename'

# read the data and build a list of pairs (table_ids and the corresponding 
# hypothesis) where the label is 'E'
df = pd.read_csv(TSV_FILE, sep='\t')
df = df.loc[df['label'].isin(['E'])]
pairs = [pair for pair in zip(list(df['table_id']), list(df['hypothesis']))]

sources, targets = [], []

# walk through the (table, hypothesis) pairs. If the hypothesis contains a 
# number, retain the pair.
for pair in pairs:
    TSV_FILE = pair[0] + '.txt'
    if TSV_FILE in os.listdir(TABLE_DIR):
        with open(os.path.join(TABLE_DIR, TSV_FILE), 'r', 
            encoding='utf-8') as fd:
            source = fd.read()
            target = pair[1]        
            if re.search(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+',
                target):            
                sources.append(source)
                targets.append(pair[1])

# write .source and .target files        
with open(OUTPUT_FILENAME + '.sources', 'w', encoding='utf-8') as fd_sources, \
    open(OUTPUT_FILENAME + '.targets', 'w') as fd_targets:

    for pair in zip(sources, targets):
        fd_sources.write(pair[0]+ '\n')
        fd_targets.write(pair[1] + '\n')
