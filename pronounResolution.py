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
    {char: [A, B, ...]} (using a list to handle possible issue with plural pronouns
    '''
    hasChar = False
    for token in rows.tokens:
        #print(token)
        if token['pos'] == 'PRON':
            token['char'] = [np.random.choice(cList)]
            hasChar = True
            
    return rows.tokens, hasChar

def pronResolution_sent(cDict, rows):
    '''
    cDict is a dictionary of characters and their total sentiment values in the movie
    '''
    hasChar = False
    return rows.tokens, hasChar

def pronEval(dfList, num=50):
    '''
    This function takes a list of dfs and evaluate model performance
    Enhance the original google api df with one of the model functions
    '''
    numModels = len(dfList)
    sampled = np.zeros(numModels)
    correct = np.zeros(numModels)
    
    for i in range(num*numModels):
        selectModel = np.random.choice(range(numModels))
        
        df = dfList[selectModel]
        charIndex = list(df[df.hasChar == True].index)
        selectLine = np.random.choice(charIndex)
        charList = [(x['content'], x['char']) for x in df.loc[selectLine]['tokens'] if 'char' in x]
        print('*******for the following dialogue********')
        for rowNum in range(max(0, selectLine - 2), min(len(dfList[selectModel]), selectLine + 3)):
            if rowNum == selectLine:
                print('=>{}: {}<='.format(df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
            else:
                print('{}: {}'.format(df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
        print('*'*20)
        print('{} pronouns resolved'.format(len(charList)))
        for i, char in enumerate(charList):
            print('{}. {} => {}'.format(i+1, char[0], char[1]))
        
        collectInput = False
        
        while not collectInput:
            try:
                count = int(input('how many are correctly identified?'))
                collectInput = True
            except:
                print('incorrect input, only numbers allowed')
        
        sampled[selectModel] += len(charList)
        correct[selectModel] += count

    print('*'*20)
    for i, modelResult in enumerate(zip(correct, sampled)):
        if modelResult[0] == 0:
            result = 0
        else:
            result = modelResult[0]/modelResult[1]
        print('model {} precision: {}'.format(i+1, result))