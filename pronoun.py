import re
import nltk
import numpy as np
import pandas as pd
import os

def pronResolution(cList, tokens):
    '''
    cList is a list of characters(str) that appear in the movie
    tokens are from the processed csv files
    expect the function to add a char tag to pronouns of interest, 
    either matching it to a name directly in cList, or another reasonable entity
    '''
    for token in tokens:
        #print(token)
        if token['pos'] == 'PRON':
            token['char'] = 'narrator'
    return tokens