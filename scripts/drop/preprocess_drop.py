'''
    preprocess_drop.py - this script preprocesses the original drop data to be usable for finetuning in our project
'''

import json

def get_answer(answer:dict) -> str:
    '''
        this method brings the answer into a interpretable shape
        :param answer: the answer dictionary that contains the answer in one 
            of its fields
        :return: formatted str of the answer
    '''
    if len(answer['number']) > 0:
        return answer['number']
    elif len(answer['spans']) > 0:
        return answer['spans'][0]
    else:
        return str(answer['date']['day'] + ' ' + answer['date']['month']\
            + ' ' + answer['date']['year']).strip()
    
# read DROP json files
DROP_JSON = '/path/to/drop_dataset/drop_dataset_dev.json'
with open(DROP_JSON, 'r', encoding='utf-8') as fd:
    data = json.load(fd)

# reformat DROP data
sources, targets = [], []
for value in data.values():    
    for question in value['qa_pairs']:        
        sources.append(value['passage'] + ' </s> ' + question['question'])
        targets.append(get_answer(question['answer']))

with open('dev.source', 'w') as source, open('dev.target', 'w') as target:
    [source.write(sample + '\n') for sample in sources]
    [target.write(sample + '\n') for sample in targets]
