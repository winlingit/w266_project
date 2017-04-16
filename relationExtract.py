import re
import nltk
import numpy as np
import pandas as pd
import os

def simpleRE(tokens):
    '''
    takes in tokens and extract potential relations
    if 'char' is available for a token, use the value instead of the actual content
    '''
    relation = []
    nsubj = -1
    verb = -1
    for i, token in enumerate(tokens):
        if token['label'] == 'NSUBJ':
            nsubj = i
            verb = token['index']
        if 'OBJ' in token['label'] and token['index'] == verb:
            #print(tokens[nsubj])
            subj = tokens[nsubj].get('char', tokens[nsubj]['content'])
            obj = tokens[i].get('char', tokens[i]['content'])
            relation.append({'verb':tokens[verb]['content'], 'noun':subj, 'obj':obj})
        
    if relation:
        return relation
    else:
        return None