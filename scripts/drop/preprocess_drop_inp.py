'''
    preprocess_drop_inp.py - this script preprocesses the original drop data to be usable in the inferable number prediction task
'''

import re
import json
from typing import List, Tuple
from nltk import pos_tag, sent_tokenize, word_tokenize

def preprocess(text:str) -> Tuple[list, list]:
    '''
        preprocess passed text, tokenize into sentences, words, and
        identify pos_tags
        :param text: str repr of text to preprocess
        :return: Tuple[list, list] with list of tokenized sentences 
            and pos tags
    '''
    sents = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sents]
    pos_tags = [pos_tag(word) for word in words]    
    return sents, pos_tags


def search_nums(text:str) -> List:
    '''
        search the passed text for numbers
        :param text: str text to search for numbers
        :return: List of unique identified numbers
    '''
    nums = re.findall(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+', text)
    return list(set(nums))


# read the drop json file
DROP_JSON = '/path/to/drop_dataset/drop_dataset_dev.json'
with open(DROP_JSON, 'r', encoding='utf-8') as fd:
    data = json.load(fd)

# process the data
sources, targets = [], []
for value in data.values():    
    for question in value['qa_pairs']:

        # preprocess passage and questions and collect relevant pos tags
        # that indicate entities
        sents_para, pos_tags_para = preprocess(value['passage'])
        pos_tags_para = [[tag[0] for tag in tags 
            if tag[1] in ['NN', 'NNP', 'NNS', 'CD']] for tags in pos_tags_para]
        sents_q, pos_tags_q = preprocess(question['question'])
        pos_tags_q = [tag[0] for tags in pos_tags_q 
            for tag in tags if tag[1] in ['NN', 'NNP', 'NNS', 'CD']]

        # reduce paragraph to sentences that share an entity with 
        # the questions
        new_para = []
        for i, tags in enumerate(pos_tags_para):
            if len(list(set(tags) & set(pos_tags_q))) > 0:
                new_para.append(sents_para[i])
        new_para = ' '.join(new_para)

        # check the paragraph + question are not longer than 1024,
        # so that the sequence can be processed by BART and T5
        if len(new_para + '</s> ' + question['question']) <= 1024:

            # search for numbers in the reduced paragraph and the 
            # question
            nums_pass = search_nums(new_para)
            nums_question = search_nums(question['question'])

            # if there are any overlapping numbers, add the sample to the 
            # dataset
            if len(list(set(nums_pass) & set(nums_question))) > 0:
                sources.append(new_para)
                targets.append(question['question'])

with open('dev.source', 'w') as source, open('dev.target', 'w') as target:
    [source.write(sample + '\n') for sample in sources]
    [target.write(sample + '\n') for sample in targets]
