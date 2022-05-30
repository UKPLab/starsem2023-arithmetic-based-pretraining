'''
    calc_parent.py - This is the script for calculating the parent score
'''

from parent import parent
import json

# The file with tables in parent format
PARENT_FILE = '/path/to/parent/file.jl'
# The file with target values
TARGET_FILE = '/path/to/target/file.txt'
# The file with predictions
PREDICTIONS = '/path/to/predictions/file.txt'

with open(PARENT_FILE, mode="r", encoding='utf8') as pf, \
    open(TARGET_FILE, mode="r", encoding='utf8') as tf, \
    open(PREDICTIONS, mode="r", encoding='utf8') as f:

    _tables = [json.loads(line) for line in pf if line.strip()]
    references = [line.strip().split() for line in tf if line.strip()]
    predictions = [line.strip().split() for line in f if line.strip()]

tables = [[tuple(attr_val) for attr_val in table] for table in _tables]

precision, recall, f_score = parent(
    predictions,
    references,
    tables,
    avg_results=True
)

print('Parent - Precision: %s' % precision)
print('Parent - Recall: %s' % recall)
print('Parent - f_score: %s' % f_score)