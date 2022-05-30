from moverscore_v2 import get_idf_dict, word_mover_score
from nltk.translate import meteor_score
import sacrebleu as scb
import os
import numpy as np
import torch
import string
from utils import is_float
from collections import Counter
from typing import List
import re

def eval_mover_score(ref_file, pred_file):

    try:
        refs = get_lines(ref_file)
        sys = get_lines(pred_file)

        idf_dict_hyp = get_idf_dict(sys) 
        idf_dict_ref = get_idf_dict(refs) 

        scores = word_mover_score(refs, sys, idf_dict_ref, idf_dict_hyp, \
                          stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)       

        return round(np.mean(scores),3) , round(np.median(scores),3 )
    except Exception as e:        
        print('exception on mover score')
        print(e)
        return 0, 0
    
def get_lines(fil):
    lines = []
    with open(fil, 'r', encoding='utf-8') as f:
        lines = [line.replace('\n', '').strip() for line in list(f)
                if len(line) > 1]
        #for line in f:
        #    if line.strip():
        #        lines.append(line.strip())
        #    else:
        #        lines.append('empty')
    return lines

def eval_sacre_bleu(ref_file, pred_file):
    try:
        refs = [get_lines(ref_file)]
        sys = get_lines(pred_file)
        bleu = scb.corpus_bleu(sys, refs)
        return bleu.score
    except:
        return 0

def em_score(prediction:List[str], target:List[str]):
    res = 0.0
    for i, pred in enumerate(prediction):
        res += (_normalize_answer(pred) == _normalize_answer(target[i]))
    return round(res / len(prediction), 3)

def em_score_drop(a_pred_file, a_gold_file):
    '''
        implementation from DROP evaluation script (https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py, last accessed 12.05.2022)
    '''

    a_gold = get_lines(a_gold_file)
    a_pred = get_lines(a_pred_file)

    #print('em_score pred: ' + str(a_pred))
    #print('em_score gold: ' + str(a_gold))

    res = 0.0
    for i, pred in enumerate(a_pred):
        res += (_normalize_answer_drop(pred) == _normalize_answer_drop(a_gold[i]))
    return round(res/len(a_pred), 3)

def _normalize_answer(text:str):
    """Lower text and remove punctuation, articles and extra whitespace."""
    
    # lowercase
    text = text.lower()

    # remove punctuation (except for numbers)
    text = ' '.join(word.strip(string.punctuation) if not is_float(word) else word for word in text)

    # fix whitespace and return
    return ' '.join(text.split())

def _normalize_answer_drop(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
      regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
      return re.sub(regex, ' ', text)
    def white_space_fix(text):
      return ' '.join(text.split())
    def remove_punc(text):
      exclude = set(string.punctuation)
      return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
      return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens_drop(s):
    if not s: 
        return []
    return _normalize_answer_drop(s).split()

def f1_score_drop(a_pred_file, a_gold_file):

    '''
        f1_score with drop-like token normalization
    '''

    a_gold = get_lines(a_gold_file)
    a_pred = get_lines(a_pred_file)

    #print('f1_score pred: ' + str(a_pred))
    #print('f1_score gold: ' + str(a_gold))

    f1 = 0.0

    for i, pred in enumerate(a_pred):
        gold_toks = get_tokens_drop(a_gold[i])
        pred_toks = get_tokens_drop(pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            f1 += int(gold_toks == pred_toks)
            continue
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 += (2 * precision * recall) / (precision + recall)
    return round(f1/len(a_pred), 3)

    
def f1_score(prediction:List[str], target:List[str]):
    '''
        calculates a F1 score as for QA
        :param pred_embeds: predicted tokens for the masked tokens (batch size x hidden states)
        :param target_embeds: target tokens for the masked tokens (batch size x hidden states)
    '''

    f1 = 0.0

    for i, pred in enumerate(prediction):
        prediction_tokens = _normalize_answer(pred).split()
        ground_truth_tokens = _normalize_answer(target[i]).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 += (2 * precision * recall) / (precision + recall)
    return round(f1 / len(prediction), 3)
    
def eval_meteor(ref_file, pred_file):

    references = open(ref_file, 'r').readlines()
    predictions = open(pred_file, 'r').readlines()

    scores = [meteor_score.single_meteor_score(ref, pred, alpha=0.9, beta=3, gamma=0.5)
        for ref, pred in zip(references, predictions)]

    return np.mean(scores)
