import re
import nltk
import numpy as np
import pandas as pd
import os


### Model 0: Random character
def pronResolution_base(cList, row):
    '''
    cList is a list of characters(str) that appear in the movie
    tokens are from the processed csv files
    expect the function to add a char tag to pronouns of interest, 
    either matching it to a name directly in cList, or another reasonable entity
    baseline randomly associates pronouns to the list of characters
    {char: [A, B, ...]} (using a list to handle possible issue with plural pronouns
    '''
    hasChar = False
    
    # check all tokens for pronouns
    for token in row.tokens:
        # print(token)
        
        # if token is pronoun, tag with character and set character flag to True
        if token['pos'] == 'PRON':
            token['char'] = [np.random.choice(cList)]
            hasChar = True
            
    return row.tokens, hasChar


### Model 1: Adjacent speaker
def pronResolution_nn():
    pass
    '''
    I => current speaker, you => speaker before/after current speaker
    '''
    
    # if token type is "I", tag with speaker and set character flag to True
    
    # else, if token is "you", tag with speaker from previous or next dialogue
    
    # nearbyChars=np.dstack((df.speaker.shift(i).values for i in range(-2, 3)[::-1]))[0]
    # for i in range(len(df)):
    #     df.set_value(i, 'nearbyChars', nearbyChars[i])


### Model 2: Speaker n-gram model:
def pronResolution_ngram():
    pass


### Model 3: Based on transition probabilities between characters
def pronResolution_hmm():
    pass


def pronResolution_sent(cDict, rows):
    '''
    cDict is a dictionary of characters and their total sentiment values in the movie
    '''
    hasChar = False
    return rows.tokens, hasChar

def pronEval(dfList, numExamples=50):
    '''
    This function takes a list of dfs and evaluate model performance
    Enhance the original google api df with one of the model functions
    '''
    numModels = len(dfList)
    sampled = np.zeros(numModels)
    correct = np.zeros(numModels)
    
    # indexes for lines of dialogue with resolved pronouns
    df = dfList[0]
    charIndex = list(df[df.hasChar == True].index)
    
    # for each line
    for n in range(numExamples):
            
        # sample random line to evaluate resolved pronoun
        selectLine = np.random.choice(charIndex)
        
        # for each model df, select line to analyze
        for m in range(numModels):
            
            # select model results
            df = dfList[m]
            charList = [(x['content'], x['char']) for x in df.loc[selectLine]['tokens'] if 'char' in x]

            # print line being analyzed
            print('\n' + '*'*8 + ' line {} '.format(selectLine) + '*'*8)
            for rowNum in range(max(0, selectLine - 2), min(len(dfList[m]), selectLine + 3)):
                if rowNum == selectLine:
                    print('=> {}. {}:\n=> {}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
                else:
                    print('{}. {}:\n{}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))

            # print resolved pronouns from model
            print('*'*8 + ' test model {}: line {} '.format(m+1, selectLine) + '*'*8)
            print('{} pronouns resolved'.format(len(charList)))
            for i, char in enumerate(charList):
                print('{}. {} => {}'.format(i+1, char[0], char[1]))

            collectInput = False

            # prompt user for count of correctly resolved pronouns
            while not collectInput:
                try:
                    count = int(input('\nhow many are correctly identified? '))
                    collectInput = True
                except:
                    print('incorrect input, only numbers allowed')

            # update counts of lines sampled from script and correctly resolved pronouns 
            sampled[m] += len(charList)
            correct[m] += count
        
    # calculate and print precision for all models
    print('\n' + '*'*8 + ' test results ' + '*'*8)
    for i, modelResult in enumerate(zip(correct, sampled)):
        if modelResult[0] == 0:
            result = 0
        else:
            result = modelResult[0]/modelResult[1]
        print('model %i: precision = %.2f (%i/%i correct)'%(i+1, result, correct[i], sampled[i]))