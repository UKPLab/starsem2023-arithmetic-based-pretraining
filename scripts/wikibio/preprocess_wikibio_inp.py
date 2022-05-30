'''
    preprocess_wikibio_inp.py - this script further processes the preprocessed wikibio data to be usable in the inferable number prediction task
'''

from nltk.tokenize import sent_tokenize
from os import path
from typing import List
import re

def extract_numbers(sentence:str) -> List[str]:
    '''
        extract numbers from the passed string
        :param sentence: alphanumerical sentence
        :return: list of unique numbers found in the passed sentence
    '''
    return list(set([num.strip() for num in 
        re.findall(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+', sentence)]))

def clean(s:str) -> str:
    '''
        this method cleans a passed string (removes "\n")
        :param s: string to be cleaned
        :return: cleaned string
    '''
    return s.replace('\n', '')


# path to the directory where to find the preprocessed wikibio data
WIKIBIO_PATH = '/path/to/preprocessed/wikibio'
SPLIT = 'train'

# open file descriptors for (1) the preprocessed source and target files, and 
# (2) for the files where to dump the shortened data
fd_source = open(path.join(WIKIBIO_PATH, SPLIT + '.source'), 'r',
    encoding='utf-8')
fd_target = open(path.join(WIKIBIO_PATH, SPLIT + '.target'), 'r', 
    encoding='utf-8')
fd_source_inp = open(path.join(WIKIBIO_PATH, SPLIT + '_inp.source'), 'a', 
    encoding='utf-8')
fd_target_inp = open(path.join(WIKIBIO_PATH, SPLIT + '_inp.target'), 'a', 
    encoding='utf-8')

# walk through the preprocessed source and target values, extract numbers from 
# the table, search for matching sentences in the target (and retain these 
# sentences) and put everything to the new _inp files (if the joined length is 
# <= 1024 chars so that everything can be processed and nothing is cut)
for _source, _target in zip(fd_source, fd_target):    
    table_numbers = extract_numbers(_source)
    target = ' '.join([sentence for sentence in sent_tokenize(_target)
        if any(i in table_numbers for i in extract_numbers(sentence))])

    if len(target) > 1:
        if len('<s>' + _source + '</s>' + target + '</s>') <= 1024:            
            fd_source_inp.write(clean(_source) + '\n')
            fd_target_inp.write(clean(target) + '\n')

fd_source.close()
fd_target.close()
fd_source_inp.close()
fd_target_inp.close()            
