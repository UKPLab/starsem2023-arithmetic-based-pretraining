'''
    preprocess_infotabs.py - This script uses the converted tables to create the data that can then be used for training and testing models within our project.
'''

import pandas as pd
import os


def get_label(label:str) -> str:
    '''
        This method converts the target values from characters into strings (e.g. "Entailment"). We found that this works better with text generation models like BART or T5.
        :param label: The character that represents the label
        :return: The string value that represents the label (e.g. "Entailment" 
            for "E")
    '''

    if 'E' in label:
        return 'Entailment'
    elif 'C' in label:
        return 'Contradiction'
    elif 'N' in label:
        return 'Neutral'
    else:
        raise Exception('label not processable')

# This is the tsv file for the data split (e.g. "infotabs_test_alpha3.tsv" for 
# the third test split)
TSV_FILE = '/path/to/tsv/file.tsv'

# This is the directory that contains the tables in linearized representation
TABLE_DIR = '/path/to/tables/preprocessed'

# This is the name of the output file (e.g. "test3" -> "test3.source", "test3.
# target")
OUTPUT_FILENAME = 'output_filename'

# read the tsv file and collect triples with table_id, hypothesis and labels 
# for all samples
df = pd.read_csv(TSV_FILE, sep='\t')
triples = [pair for pair in zip(list(df['table_id']), list(df['hypothesis']),
    list(df['label']))]

files = os.listdir(TABLE_DIR)
sources, targets = [], []

# walk through the triples, read the preprocessed table (using the table_id), 
# concat it with the corresponding hypothesis (triple[1]), get the label 
# (triple[2]) and add them to the source and target lists.
for triple in triples:

    filename = triple[0] + '.txt'
    if filename in files:
        with open(os.path.join(TABLE_DIR, filename), 'r', 
            encoding='utf-8') as fd:
        
            # if you want to experiment with prefixes in T5, you could add 
            # them here using this line of code:
            # source = "mnli hypothesis: " + fd.read() + '. premise: '\
            #  + triple[1]
        
            sources.append(fd.read() + ' </s> ' + triple[1])
            targets.append(get_label(triple[2]))

# write .source and .target files
with open(OUTPUT_FILENAME + '.source', 'w', encoding='utf-8') as source, \
    open(OUTPUT_FILENAME + '.target', 'w', encoding='utf-8') as target:

    for pair in zip(sources, targets):
        source.write(pair[0]+ '\n')
        target.write(pair[1] + '\n')
