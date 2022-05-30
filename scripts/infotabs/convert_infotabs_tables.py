'''
    convert_infotabs_tables.py -- This script converts the tables for infotabs (which are delivered separately in json files) into linearized represention which can be used in this project.
'''

import json
import os
from typing import Dict

def process(tab:Dict, dir:str, filename:str) -> None:
    '''
        This method extracts keys and values from the table in json format and incorporates them into a linearized representation which is then stored as txt file in a "preprocessed" subdirectory.
        :param tab: the table in json format
        :param dir: the parent directory where to store the tables
        :param filename: the name of the table's json file
        :return: None
    '''

    # keys - table headers
    # values - table values
    keys = list(tab.keys())    
    values = [', '.join(value_list) if len(value_list) > 1 else value_list[0] 
        for value_list in list(tab.values())]
    
    table = '<R> <C> ' + ' <C> '.join(keys) + ' <R> <C> ' \
        + ' <C> '.join(values)
    
    dir = os.path.join(dir, 'preprocessed')
    if not os.path.exists(dir):
        os.mkdir(dir)

    # write the table into a textfile that is named after the table_id
    with open(os.path.join(dir, filename.split('.')[0] + '.txt'), 
        'w', encoding='utf-8') as fd:
        fd.write(table)
    
DIRECTORY = '/path/to/infotabs/data/tables/json'

# go through the table directory and convert all json tables into linearized 
#  representations
for file in os.listdir(DIRECTORY):
    filename = os.path.join(DIRECTORY, file)    
    if os.path.isfile(filename):        
        with open(filename, 'r', encoding='utf-8') as fd:
            json_tab = json.load(fd)        
            process(json_tab, DIRECTORY, file)
