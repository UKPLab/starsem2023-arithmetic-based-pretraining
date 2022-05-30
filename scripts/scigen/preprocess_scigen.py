'''
    preprocess_scigen.py -- This script preprocesses the scigen data (which is delivered in json files) into linearized represention which can be used in this project.
'''

from os import path
import json
import re

# path to the folder with the preprocessed scigen files
SCIGEN_PATH = '/path/to/scigen'
# filenames
SCIGEN_FILE = path.join(SCIGEN_PATH, 'dev.json')
SOURCE_FILE = path.join(SCIGEN_PATH, 'dev.source')
TARGET_FILE = path.join(SCIGEN_PATH, 'dev.target')

# load the scigen data
with open(SCIGEN_FILE, 'r', encoding='utf-8') as file:
    tables = json.load(file)

# open a file descriptor for the source and target files and walk through all 
# scigen tables, build a linearized representation and dump it into the files
with open(SOURCE_FILE, 'w', encoding='utf8') as fd_source, \
    open(TARGET_FILE, 'w', encoding='utf8') as fd_target:
    for table in tables.values():
        linearized = '<R> '
        for col_header in table['table_column_names']:
            linearized += ' <C> ' + col_header
        for row in table['table_content_values']:
            linearized += ' <R> <C> ' + ' <C> '.join(row)
        linearized += ' <CAP> ' + table['table_caption']

        fd_source.write(re.sub(' +', ' ', linearized) + '\n')
        fd_target.write(table['text'] + '\n')
    