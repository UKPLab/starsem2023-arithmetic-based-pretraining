import re
import inflect
import os
from tqdm import tqdm
import json
import torch
from typing import Tuple
import logging
from random import randrange
from char_level_representation import char_level_representation_encoding, pad_or_trim
from scientific_notation import rebuild_string_in_new_notation
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

logger = logging.getLogger(__name__)
logging.basicConfig(filename='test.txt', level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

handler_char_level_representation = logging.FileHandler('char_level_representation_output.txt')
file_logger_dt = logging.getLogger('file_logger_dt')
file_logger_dt.addHandler(handler_char_level_representation)

handler_scientific_notification = logging.FileHandler('scientific_notifiation_output.txt')
file_logger_sn = logging.getLogger('file_logger_sn')
file_logger_sn.addHandler(handler_scientific_notification)

def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def trim_batch(input_ids, pad_token_id, attention_mask=None, trim_pos=None):
    '''
        remove columns that are populated exclusively by pad_token_id
        :param input_ids: ids where to remove the pad token
        :param pad_token_id: id of the pad token
        :param attention_mask: attention mask for the input_ids (optional)
        :param trim_pos: list of lengths for the original input text; if 
            passed, all input_ids of the batch are trimmed to the max value
            of this list
        :return: batch of trimmed input_ids (and attention masks if they were passed)
    '''

    if trim_pos is not None:
        return input_ids[:, :trim_pos] if attention_mask is None \
            else (input_ids[:, :trim_pos], attention_mask[:, :trim_pos])
    else:
        # boolean 'mask' of values to keep or to remove
        column_mask = input_ids.ne(pad_token_id).any(dim=0)

        return input_ids[:, column_mask] if attention_mask is None \
            else (input_ids[:, column_mask], attention_mask[:, column_mask])

#def convert_text(tokens:str) -> str:
    '''
        this method cleans the passed string after generation   
        :param text: list of tokens
        :return: cleaned string
    '''    
#    text = ' '.join(re.split('(\W)', tokens))
#    text = ' '.join(text.split())
#    return text.lower()

def rnd_logging(file_logger, log_dict:dict):
    '''
        for manually checking whether the notations are correctly applied,
        we want to save some input_ids and attention_masks (but not all
        of them)
    '''

    # we need samples to check that implementation is correct
    # we don't want to log every sample...
    if randrange(100) % 5 == 0:
        for key, value in log_dict.items():
            file_logger.info(key + ':')
            file_logger.info(value)

def encode_file_bart(tokenizer, data_path, max_length, scientific_notation, char_level_representation, pad_to_max_length=True, 
    return_tensors="pt"):

    examples = []
    with open(data_path, "r") as f:
        for text in tqdm(f.readlines()):

            text = text.strip()

            logger.info(data_path)

            if char_level_representation:
                tokenized = char_level_representation_encoding(text, tokenizer, max_length)    

            else:
                if scientific_notation:
                    logger.info('In scientific tokenization')
                    text_new = rebuild_string_in_new_notation(text)              
                else:
                    text_new = text

                tokenized = tokenizer.batch_encode_plus(
                    [text_new], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors, #add_special_tokens=True,
                )

                rnd_logging(file_logger_sn, {'input_ids': tokenized['input_ids'], 
                    'attention_mask': tokenized['attention_mask']})

            examples.append(tokenized)
            
    return examples

def generate_masked_sequence(source_str, target_str, token, tokenizer, max_length_source, max_length_target,
    char_level_representation=False, scientific_notation=False):
    '''
        returns the sequence tokenized with the corresponding tokenization algorithm, the length of the target sequence (in tokens) and the 
        position of the masked tokens in the target.
    '''

    if char_level_representation:

        # encode everythin in digit tokenization
        _source = char_level_representation_encoding(source_str, tokenizer, max_length_source)
        target = char_level_representation_encoding(target_str, tokenizer, max_length_target)
        _token = char_level_representation_encoding(token, tokenizer, len(token))

        # now mask the spans of the token to be masked in the target
        tokenized_target = ' '.join(target['tokenized'])
        tokenized_token = ' '.join(_token['tokenized'][1:-1])                
        mask = ' '.join(['<mask>' for i in range(0, len(_token['tokenized'][1:-1]))])                
        masked_target = tokenized_target.replace(tokenized_token, mask).split()

        tokenized_target.replace(tokenized_token, mask)

        if isinstance(tokenizer, T5TokenizerFast):
            tokenized_token = ' '.join(_token['tokenized'][:-1])
            mask = ' '.join(['<extra_id_' + str(i) + '>' for i in range(0, len(_token['tokenized'][:-1]))])
            masked_target = tokenized_target.replace(tokenized_token, mask).split(' ')
        else:
            tokenized_token = ' '.join(_token['tokenized'][1:-1])
            mask = ' '.join(['<mask>' for i in range(len(_token['tokenized'][1:-1]))])
            masked_target = tokenized_target.replace(tokenized_token, mask)\
                .split()
            tokenized_target.replace(tokenized_token, mask)   

        # create the complete input sequence (table + cap + masked target)
        sequence = _source['tokenized'] + masked_target[1:]

        # adjust the splitted indices, add the length of the source table + cap to the splitted
        # indices of the target and concat both lists
        target_splitted_indices = [(idx[0] + len(_source['tokenized']) -1, idx[1] + len(_source['tokenized']) -1) 
            for idx in target['splitted_indices']]        
        source_splitted_indices = [(idx[0], idx[1]) for idx in _source['splitted_indices']]
        splitted_indices = source_splitted_indices + target_splitted_indices

        # neede to delete those spanse from the contrastive example here
        if isinstance(tokenizer, T5TokenizerFast):
            indices_to_delete = [i for i, idx in enumerate(splitted_indices) if '<extra_id_' in sequence[idx[0]:idx[1]]]
        else:
            indices_to_delete = [i for i, idx in enumerate(splitted_indices) if '<mask>' in sequence[idx[0]:idx[1]]]

        # convert the input sequence into input ids and generate the attention mask
        source_input_ids, source_attention_mask =\
            pad_or_trim(torch.LongTensor(tokenizer.convert_tokens_to_ids(sequence)),
                max_length_source, torch.LongTensor([2]), torch.LongTensor([1]))

        # make sure that there are no splitted indices in the list that aren't in the input sequence
        # anymore
        source_splitted_indices = [indice for indice in splitted_indices if indice[1] < 
            max_length_source]
        
        # the same for the target
        target_splitted_indices = [(idx[0]+1, idx[1]+1) for idx in target['splitted_indices']]
        target['splitted_indices'] = target_splitted_indices

        source = {'input_ids': source_input_ids.unsqueeze(0), 
                  'attention_mask': source_attention_mask.unsqueeze(0), 
                  'splitted_indices': source_splitted_indices,
                  'indices_to_delete': indices_to_delete}        

        if isinstance(tokenizer, T5TokenizerFast):
            masked_ids_target = [i for i in range(0, len(masked_target[1:])) if '<extra_id_' in masked_target[i] and i < max_length_source]
        else:
            masked_ids_target = [i for i in range(0, len(masked_target[1:])) if masked_target[i] == '<mask>' and i < max_length_source] 
    
    else:

        _source = tokenizer.tokenize(source_str)
        _target = tokenizer.tokenize(target_str)
        _token = tokenizer.tokenize(token)

        token_seq = ' '.join(_token)

        mask = ' '.join(['<mask>' for i in range(0, len(_token))])
        masked_target = re.sub(r'Ġ'+token_seq + '|' + token_seq, mask, ' '.join(_target)).split()

        if '+' in token_seq:
            token_seq = token_seq.replace('+', '\+')
        if '-' in token_seq:
            token_seq = token_seq.replace('-', '\-')

        if not isinstance(tokenizer, T5TokenizerFast):
            masked_target = re.sub(r'Ġ'+token_seq + '|' + token_seq, mask, ' '.join(_target)).split()
        else:
            masked_target = re.sub(r'▁'+token_seq + '|' + token_seq, mask, ' '.join(_target)).split()

        if not isinstance(tokenizer, T5TokenizerFast):
            sequence = ['<s>'] + _source + ['</s>'] + masked_target + ['</s>']
        else:
            sequence = _source + ['</s>'] + masked_target + ['</s>']
        
        # convert the input sequence into input ids and generate the attention mask
        source_input_ids, source_attention_mask =\
            pad_or_trim(torch.LongTensor(tokenizer.convert_tokens_to_ids(sequence)),
                max_length_source, torch.LongTensor([2]), torch.LongTensor([1]))    
        target = tokenizer.batch_encode_plus([target_str], max_length=max_length_target, pad_to_max_length=True, 
            return_tensors='pt')     

        source = {'input_ids': source_input_ids,
                  'attention_mask': source_attention_mask}            

        masked_ids_target = [i+1 for i in range(0, len(masked_target)) if masked_target[i] == '<mask>']

    if len(masked_ids_target) > 0:
        masked_ids_target.append(masked_ids_target[-1]+1)

    return source, target, len(masked_target), masked_ids_target

def encode_scigen_mask_nums(tokenizer, data_path_source, data_path_target, max_length_source, 
    max_length_target, scientific_notation=False, char_level_representation=False, contrastive=False, verbalized=False):

    file_source = open(data_path_source, 'r', encoding='utf-8')
    file_target = open(data_path_target, 'r', encoding='utf-8')
    data_pairs = zip(file_source.readlines(), file_target.readlines())
    file_source.close()
    file_target.close()

    samples = []
    pairs = []
    
    for (source, target) in tqdm(data_pairs):

        # find potential numbers for masking from all rows except the first one
        # capture also zero or one +/- sign (to make sure such numbers are also found)
        pot_mask_nums = list(set(re.findall(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+', target)))
        
        # if you want to do the ablation with also masking words:
        #pot_mask_nums = list(set(re.findall(r'(?<!\w)[\+\-]{0,1}\d*[\.]{0,1}\d+|\w+', target)))
        
        for token in pot_mask_nums:

            if char_level_representation:  

                char_level_sample, char_level_target, masked_target_length, masked_ids_target =\
                    generate_masked_sequence(source, target, token, tokenizer, max_length_source, max_length_target, 
                        char_level_representation=char_level_representation)
                
                if len(char_level_sample['splitted_indices']) == 0:
                    continue
                                                
                sample = {'source': char_level_sample,
                          'target': char_level_target,
                          'masked_ids_target': masked_ids_target,
                          'length_masked_target': masked_target_length,
                          'original': source
                        }                
                samples.append(sample) 

                sample['source']['splitted_indices'] = [idx for i, idx in 
                    enumerate(sample['source']['splitted_indices']) if i 
                    not in sample['source']['indices_to_delete']]

                if contrastive:
                    
                    # this is okay in case since it is only important that the number to be masked 
                    # is not used for calculating the contrastive loss
                    #sequence = source + '</s>' + re.sub(r' ', ' ', 
                    #    re.sub(r'\b' + token + r'\b', ' <mask> ', target))
                    sequence = source + '</s>' + target
                                             
                    # masked_ids_target und length_masked_target are not necessary here,
                    # but need to be passed in order for the collate function to work properly
                    sample = {'source': get_contrastive_sample(sequence, tokenizer, 
                                            max_length_source, verbalized),
                              'target': get_contrastive_sample(target, tokenizer, 
                                            max_length_target, verbalized),       
                              'masked_ids_target': masked_ids_target,
                              'length_masked_target': masked_target_length,
                              'original': source
                        }  

                    sample['source']['splitted_indices'] = [idx for i, idx in 
                        enumerate(sample['source']['splitted_indices']) if i 
                        not in samples[-1]['source']['indices_to_delete']
                        and i < max_length_source]
                    sample['target']['splitted_indices'] = [idx for i, idx in 
                        enumerate(sample['target']['splitted_indices'])
                        if i < max_length_target]                        

                    samples.append(sample)
                    pairs.append((len(samples)-2, len(samples)-1))
                
            else:
                if scientific_notation:
                    source = rebuild_string_in_new_notation(source) 
                    target = rebuild_string_in_new_notation(target)   
                    token =  rebuild_string_in_new_notation(token)   

                _source, _target, masked_target_length, masked_ids_target =\
                    generate_masked_sequence(source, target, token, tokenizer, max_length_source, max_length_target, 
                        char_level_representation=char_level_representation)

                sample = {'source': _source,
                    'target': _target,
                    'masked_ids_target': masked_ids_target,
                    'length_masked_target': masked_target_length,
                    'original': source}
                
                samples.append(sample)  

    if len(samples) == 0:
        logger.error('No masked samples could be extracted from the passed data files!')
        exit()
    
    return samples, pairs  

def get_contrastive_sample(orig_sequence, tokenizer, max_length, verbalized):      

    classic_text, num_spans = preprocess_contrastive(orig_sequence, tokenizer, only_number if not verbalized else verbalize)
    
    input_ids_classic, attention_mask_classic = pad_or_trim(torch.LongTensor(
        tokenizer.convert_tokens_to_ids(classic_text)), max_length, torch.LongTensor([2]), 
        torch.LongTensor([1]))
    
    num_spans = [span for span in num_spans if span[1] < max_length]

    return {'input_ids': input_ids_classic.unsqueeze(0), 
            'attention_mask': attention_mask_classic.unsqueeze(0), 
            'splitted_indices': num_spans}

def verbalize(num:str):   
    p = inflect.engine()
    num = str(num.group(0)).strip()

    if '.' in num:        
        exp, mant = num.split('.')
        if not is_float(exp):
            exp = '0'
        if not is_float(mant):
            mant = '0'
        return '[F] '+ p.number_to_words(exp) + ' point ' + p.number_to_words(mant) + ' [F]'
    else:
        return '[F] '+ p.number_to_words(num) + ' [F]'

def only_number(num:str):       
    num = str(num.group(0)).strip()
    return '[F] '+ num + ' [F]'

def preprocess_contrastive(seq:str, tokenizer, func):            

    seq = '<s>' + seq + '</s>'

    seq = re.sub(r' +', ' ', re.sub(r'[+-]?(\d*\.)?\d+', func, seq))

    tokenized = tokenizer.tokenize(seq)
    num_spans = []
    num_indices = [i for i, _ in enumerate(tokenized) if _ == '[F]']
    num_indices = list(zip(num_indices[::2], num_indices[1::2]))
    if len(num_indices) > 0:
        result = tokenized[:num_indices[0][0]]

        for i, num_indice in enumerate(num_indices):
            number = tokenized[num_indice[0]+1:num_indice[1]]
            num_spans.append((len(result), len(result) + len(number) - 1))
            result.extend(number)

            if i+1 < len(num_indices):
                result.extend(tokenized[num_indice[1] + 1: num_indices[i+1][0]])

        result.extend(tokenized[num_indices[-1][1] + 1:])
    
    else:
        result = tokenized 

    return result, num_spans

def visualize_embeddings(embeds_1, embeds_2):
    

    dim_reducer = TSNE(n_components=2)
    fig, ax = plt.subplots(1, 1, figsize = (6, 6), dpi=300)

    labels = ['Char-Level Tokenization' for i in range(0, embeds_1.size()[0])]\
         + ['Subword Tokenization' for i in range(0, embeds_2.size()[0])]
    embeds = torch.cat((embeds_1, embeds_2))
    reduced = dim_reducer.fit_transform(embeds.numpy())
    df = pd.DataFrame.from_dict({'x':reduced[:,0],
        'y':reduced[:,1]})

    sns.scatterplot(data=df,x='x',y='y',hue=labels)
    
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('With Contrastive Learning')

    plt.savefig(f'number_embeddings',format='png',pad_inches=0)
