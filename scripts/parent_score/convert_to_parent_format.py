'''
    convert_to_parent_format.py - This script converts linearized tables into 
    the (attribute, values) format needed for calculating the PARENT score 
    (https://github.com/KaijuML/parent)
'''

import json
from typing import List
import re

def _preprocess(text:str) -> List[str]:
    '''
        This method preprocesses the passed linearized table in that it 
        removes all additional tags (e. g. [BOLD] or < bold>, but remain <C>), which are not part of the target, and everything that is not 
        A-Za-z0-9<> (keep <> for the special char <C>). Further, it removes 
        all whitespaces
        :param text: text to be preprocessed
        :return: the preprocessed string as a list of words
    '''
    
    preprocessed = re.sub(r' +', ' ', re.sub(r'(\[\w+\]|<[^C]\w+>|< +[^C]\w+>|<[^C]\w+ +>)', r'', text))
    
    # if it's not possible to clean the string without removing everything, 
    # just keep it...
    if len(preprocessed) == 0:
        preprocessed = text
    return preprocessed.split()

def _preprocess_list(text:List[str]) -> List[List[str]]:
    '''
        This method preprocesses list of strings by using _preprocess
        :param text: list of strings (or text sequences)
        :return: list of preprocessed text sequences
    '''
    values = []
    if isinstance(text, str):
        return _preprocess(text)
    else:
        [values.extend(_preprocess(value)) for value in text]    
        return values

# source and output file
SOURCE_FILE = '/path/to/source/file.source'
OUTPUT_PARENT_FILE = '/path/to/target/parent/file.jl'

# read source data
with open(SOURCE_FILE, 'r', encoding='utf-8') as file:
    source_data = [line for line in file]

# split table and caption
tables = [line.split('<CAP>')[0].split('<R>')[1:] for line in source_data]

# get table col-wise
tables_with_cols = [[line.split('<C>')[1:] for line in table] for table 
    in tables]

# get the column names
column_names = [table[0] + ['CAP'] for table in tables_with_cols] 

# walk through the table and extract the values
column_values = []
for j, table in enumerate(tables_with_cols):
    vals = [[table[k][i] for k in range(1, len(table))] 
        for i in range(len(table[0]))]
    vals += [source_data[j].split('<CAP>')[1]]    
    column_values.append(vals)

# preprocess the values and convert everything to the parent format
converted_data = [[[_preprocess(column_name[i]), 
    _preprocess_list(column_value[i])] for i in range(len(column_name))] 
        for column_name, column_value in zip(column_names, column_values)]

# create the parent file
with open(OUTPUT_PARENT_FILE, 'a', encoding='utf-8') as file:
    for line in converted_data:
        json.dump(line, file)
        file.write('\n')