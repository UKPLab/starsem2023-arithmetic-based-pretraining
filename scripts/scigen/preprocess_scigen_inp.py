'''
    preprocess_scigen_inp.py - this script further processes the preprocessed scigen data to be usable in the inferable number prediction task
'''

import os
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import re
import tqdm

def is_num(num) -> bool:
    '''
        This method checks whether an arbitrary passed value is a floating 
        point number
        :param num: an arbitrary value
        :return: whether the passed value is a floating point number
    '''
    try:
        float(num)
        return True
    except ValueError:
        return False

def read_file(file:str) -> List[str]:
    '''
        This method reads the file to the passed filename linewise and returns
        the data
        :param file: the name of the file
        :return: the data read from the file (linewise)
    '''    
    with open(file, 'r', encoding='utf-8') as fd:
        return [line for line in fd]    
    
def split_table(table:str) -> Tuple[List[str], str]:
    '''
        This method separates the table from its caption, and converts the table into a "matrix" for easier access
        :param table: the table in linearized representation
        :return: the table as matrix and its caption
    '''
    rows, cap = table.split('<CAP>')
    rows = rows.split('<R>')[1:]
    table_matrix = [row.split('<C>')[1:] for row in rows]
    return table_matrix, cap

def clean_string(text:str) -> str:
    '''
        This method cleans a passed string. (1) it removes all additional tags 
        (e. g. [BOLD] or <bold>, but retain <C>), which are not part of the target, and (2) it removes all multi whitespaces
        :param text: text to be cleaned
        :return: the cleaned text
    '''    
    regex = '(\[\w+\]|<[^C]\w+>|< +[^C]\w+>|<[^C]\w+ +>)'
    return re.sub(r' +', ' ', re.sub(fr'{regex}', r'', text))

def get_table_entities(table:List[List[str]]) -> List[str]:     
    '''
        This method collects all values from the first two rows and from the first two columns (these are most likely row and column headers)
        :param table: the table matrix
        :return: a list of unique row and cloumn headers
    '''    
    raw_entities = []
    for i, row in enumerate(table):
        if i <= 1:
            raw_entities += row
        else:
            raw_entities += [row[0], row[1]]

    entities = []     
    for cell in raw_entities:    
        tokenized = nltk.word_tokenize(cell)

        # first: use ne_chunk (will not find everything...); will also find 
        # compound nouns
        for chunk in nltk.ne_chunk(nltk.pos_tag(tokenized), binary=True):
                if type(chunk) == nltk.Tree:                
                    entities.append(" ".join([token for token, pos in chunk.leaves()]))        

        # second: complement the identified entities with those, that are 
        # entities by pos tag
        for nn in nltk.pos_tag(tokenized):
            if 'NN' in nn[1] and len(nn[0]) > 3 and nn[0] not in entities:
                entities.append(nn[0])

    return list(set(entities))

def get_values_for_entity(entity:str, table:List[List[str]]) \
    -> List[List[float]]:
    '''
        This method returns all values from the table that can be assigned to the passed entity. It collects all values from columns and rows where the entity occurs in the the header.
        :param entity: the entity
        :param table: the table in matrix form
        :return: the list of lists with all values, that can be related to 
            this entity.
    '''

    # collect all col and row names from the table
    col_names, row_names, values = [], [], []    
    for i, row in enumerate(table):
        if i == 0:
            col_names += [row]
        else:
            row_names += [row[0]]

    # collect col headers that are related to the entity
    entity_cols = [j for _cols in col_names for j, col in enumerate(_cols) 
        if entity.lower() in col.lower()]

    # collect row headers that are related to the enity
    entity_rows = [i+1 for i, row in enumerate(row_names) if entity.lower() 
        in row.lower()]

    for col in entity_cols:
        # collect all values from the column that are numeric
        _temp = [rows[col].strip() for rows in table if is_num(rows[col])]
        if len(_temp) > 0:
            values.append(_temp)

    for row in entity_rows:            
        # collect all values from the row that are numeric
        _temp = [val.strip() for val in table[row] if is_num(val)]
        if len(_temp) > 0:
            values.append(_temp)

    return values

def exact_match(num:float, values:List[float]) -> Tuple[bool, List[float]]:    
    '''
        This method checks whether there is an exact match between num and one of the values in values.
        :param num: the num for which to check
        :param values: a list of floats for comparison
        :return: a tuple describing whether there is an exact match and the 
            corresponding value
    '''
    for vals in values:        
        if num in vals:
            return True, [num]
    return False, [0.0]

def min_max(num:float, values:List[float]) -> Tuple[bool, List[float]]:
    '''
        This method checks whether the passed num is the result of an ordering operation of the values in values
        :param num: the num for which to check
        :param values: a list of floats for comparison
        :return: a tuple describing whether the passed num is the result of an 
            ordering operation
    '''
    if not exact_match(num, values):
        return False, [0.0]
    for vals in values:
        _vals = [float(v) for v in vals]        
        if float(num) in _vals:
            if float(num) == min(_vals) or float(num) == max(_vals):
                return True, [num]
    return False, [0.0]

def difference_sum(num:float, values:List[float]) -> Tuple[bool, List[float]]:
    '''
        This method checks whether the passed value is the result of a difference or sum between two values from the passed values list
        :param num: the num for which to check
        :param values: a list of floats for comparison
        :return: a tuple describing whether the passed num is the result of a 
            difference or sum operation between to values from values along with the two corresponding values
    '''
    num = round(num, 2)    
    for vals in values:
        for i in range(0, len(vals)):
            for j in range(0, len(vals)):
                if i != j:
                    if num == round(float(vals[i]) - float(vals[j]), 2) \
                        or num == round(float(vals[i]) + float(vals[j]), 2):
                        return True, [vals[i], vals[j]]    
    return False, [0.0]

def multiplication_division(num:float, values:List[float]) -> Tuple[bool, List[float]]:
    '''
        This method checks whether the passed value is the result of a multiplication or division between two values from the passed values list
        :param num: the num for which to check
        :param values: a list of floats for comparison
        :return: a tuple describing whether the passed num is the result of a 
            multiplication or division operation between to values from values along with the two corresponding values
    '''
    num = round(num, 2)
    for vals in values:
        for i in range(0, len(vals)):            
            for j in range(0, len(vals)):
                if i != j:
                    if num == round(float(vals[i]) * float(vals[j]), 2):
                        return True, [vals[i], vals[j]] 
                    try:
                        if num == round(float(vals[i]) / float(vals[j]), 2):
                            return True, [vals[i], vals[j]]
                    except ZeroDivisionError:
                        continue            
    return False, [0.0]

def shorten_table(table_matrix:List[List[str]], numbers:List[float], 
    entities:List[str], cap:str) -> str:
    '''
        This method shortens the passed table matrix to the passed numbers and entities.
        :param table_matrix: The matrix that shall be shortened in matrix 
            representation
        :param numbers: The numbers to which to shorten the table
        :param entities: The entities to which to shorten the table
        :return: The shortened table in linearized representation
    '''

    # invert tables for switching rows and columns (we want to retain the columns, this way, we can delete whole rows)
    inv_table_matrix = list(map(list, zip(*table_matrix)))

    inv_retain = [inv_table_matrix[0]]
    for i, row in enumerate(inv_table_matrix[1:]):       
        _row = ' '.join(row) 
        _nums = ' '.join([str(n).strip() for n in numbers])         
        if len([num for num in row if num.strip() in _nums]) > 0 or\
            len([num for num in numbers if str(num).strip() in _row]) > 0 or\
            len([ent for ent in entities if ent.lower() in _row.lower()]):
            inv_retain.append(row)             

    # restore the original table based on the retained values
    retain = list(map(list, zip(*inv_retain)))

    # if the restored table contain no values (except the headers), keep the original table
    if len(retain[0]) < 2:
        retain = table_matrix

    # restore the table with the retained values in linearized 
    # representationas
    return '<R>' + '<R>'.join(['<C>' + '<C>'.join(row) for row in retain]) \
        + '<CAP>' + cap if len(retain) > 0 else None


PATH = 'path/to/data/'
SOURCE_FILE = read_file(os.path.join(PATH, '/path/to/source/file.source'))
TARGET_FILE = read_file(os.path.join(PATH, '/path/to/target/file.target'))
NEW_SOURCE_FILE = os.path.join(PATH, 'new_source.source')
NEW_TARGET_FILE = os.path.join(PATH, 'new_target.target')

results = []
for source, target in tqdm.tqdm(zip(SOURCE_FILE, TARGET_FILE)):

    # clean strings
    source = clean_string(source)
    target = clean_string(target)

    # tokenize target description, create table matrix, extract table entities
    sentences = sent_tokenize(re.sub(r'Table \d+', 'Table ', target))
    table_matrix, cap = split_table(source)    
    table_entities = get_table_entities(table_matrix)

    # walk through the sentences of the target description, search for 
    # overlapping entities and contained numbers that are inferable from the 
    # table in a simple arithmetic operation or by occurence
    sents, retain_nums, retain_ents = [], [], []    
    for sentence in sentences:
        sent_entities = [entity for entity in table_entities if entity 
            in sentence]

        if len(sent_entities) > 0:
            numbers = list(set([num.strip() 
                for num in re.findall(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+', sentence)]))

            if len(numbers) > 0:
                retain_sent = False   
                for entity in sent_entities:
                
                    values = get_values_for_entity(entity, table_matrix)
                    
                    if len(values) > 0:
                        for num in numbers:
                            res = min_max(num, values)
                            if res[0]:
                                retain_nums += res[1]
                                retain_ents.append(entity)
                                retain_sent = True
                                continue                             
                            res = exact_match(num, values)
                            if res[0]:
                                retain_nums += res[1]
                                retain_ents.append(entity)
                                retain_sent = True
                                continue
                            res = difference_sum(float(num), values)
                            if res[0]:
                                retain_nums += res[1]
                                retain_ents.append(entity)
                                retain_sent = True
                                continue       
                            res = multiplication_division(float(num), values)
                            if res[0]:
                                retain_nums += res[1]
                                retain_ents.append(entity)
                                retain_sent = True
                                continue 

                if retain_sent:
                    sents.append(sentence.strip())

    if len(sents) > 0:
        source = shorten_table(table_matrix, retain_nums, retain_ents, cap)
        target = ' '.join(sents)
        
        # if source and target concatenated is <= 1024, add them to the 
        # retained data (we want to make sure that BART and T5 can see the 
        # whole sequence)
        if len(source + '</s>' + target) <= 1024:                    
            results.append((source, target))
        
with open(NEW_SOURCE_FILE, 'w', encoding='utf-8') as fd_sources, \
    open(NEW_TARGET_FILE, 'w', encoding='utf-8') as fd_targets:

    for source, target in results:
        fd_sources.write(source)
        fd_targets.write(target + '\n')
