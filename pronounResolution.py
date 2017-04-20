import re
import nltk
import numpy as np
import pandas as pd
import os

def pronResolution_base(cList, rows):
    '''
    cList is a list of characters(str) that appear in the movie
    tokens are from the processed csv files
    expect the function to add a char tag to pronouns of interest, 
    either matching it to a name directly in cList, or another reasonable entity
    baseline randomly associates pronouns to the list of characters
    '''
    for token in rows.tokens:
        #print(token)
        if token['pos'] == 'PRON':
            token['char'] = np.random.choice(cList)
    return rows.tokens