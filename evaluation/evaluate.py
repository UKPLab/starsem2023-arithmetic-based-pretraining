'''
    evaluate.py - This script can be used for manual evaluation of 
    predictions. It just calculates MoverScore and BLEU score. If you want to 
    calculate the PARENT or BLEURT score, please follow the instructions given 
    in the README.
'''

from typing import List
import numpy as np
from argparse import ArgumentParser
import os
import sacrebleu as scb
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict

def get_lines(file:str) -> List[str]:
    '''
        This method just reads the lines of the passed file
        :param file: The file to read
        :return: The content of the file, line-wise, as list
    '''
    lines = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
            else:
                lines.append('empty')
    return lines

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p", "--pred", help="prediction file", required=True)
    parser.add_argument("-s", "--sys", help="system file", required=True)

    args = parser.parse_args()
    refs = get_lines(args.sys)
    preds = get_lines(args.pred)

    bleu = scb.corpus_bleu(preds, [refs])
    print('BLEU: ', bleu.score)

    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)

    scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, \
                        stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
    print('MoverScre mean: ', np.mean(scores), 'MoverScoreMedian: ', np.median(scores))
