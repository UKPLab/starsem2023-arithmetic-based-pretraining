'''
    preprocess_wikibio.py -- This script preprocesses the wikibio data (which is delivered in json files) into linearized represention which can be used in this project.
'''

import json
from typing import List
import re

def clean(s:str) -> str:
    '''
        clean string by replacing all "\n".
        :param s: string to be cleaned
        :return: cleaned string
    '''
    return s.replace('\n', ' ')

def clean_list(l:List[str]) -> List[str]:
    '''
        clean a list of strings
        :param l: list of strings
        :return: list of cleaned strings
    '''
    return [clean(v) for v in l]    

def finalize(s:str) -> str:
    '''
        clean the passed string, fix comma issues, replace escaped "." and remove multiple whitespaces
        :param s: string to be cleaned
        :return: cleaned string
    '''
    return re.sub(r' +', ' ', re.sub(r' \.', '. ', 
        re.sub(r' ,', ', ', clean(s))))

# WikiBio source file
WIKIBIO_FILE = '/path/to/wikibio/train/train.jsonl'
# File for dumping the preprocessed source data
SOURCE_FILE = '/path/to/wikibio/train/train.source'
# File for dumping the preprocessed target data
TARGET_FILE = '/path/to/wikibio/train/train.target'

# open the file descriptors
fd_wikibio = open(WIKIBIO_FILE, 'r', encoding='utf-8')
fd_source = open(SOURCE_FILE, 'a', encoding='utf-8')
fd_target = open(TARGET_FILE, 'a', encoding='utf-8')

# for each line in the wikibio file, load its content, preprocess the table 
# stuff by cleaning and concatenation, preprocess the target text and dump 
# everything into the corresponding files
for line in fd_wikibio:
    raw_sample = json.loads(line)
    table = raw_sample['input_text']['table']

    sample = finalize('<R><C> ' 
        + ' <C> '.join(clean_list(table['column_header']))
        + ' <R><C> ' + ' <C> '.join(clean_list(table['content']))
        + ' <CAP> ' + clean(raw_sample['input_text']['context']))
    target = finalize(raw_sample['target_text'])

    fd_source.write(sample + '\n')
    fd_target.write(target + '\n')

fd_wikibio.close()
fd_source.close()
fd_target.close()    
