import re
import nltk
import numpy as np
import pandas as pd
import os


### Model 0: Random character
def pronResolution_base(charList, row):
    '''
    cList is a list of characters(str) that appear in the movie
    tokens are from the processed csv files
    expect the function to add a char tag to pronouns of interest, 
    either matching it to a name directly in cList, or another reasonable entity
    baseline randomly associates pronouns to the list of characters
    {char: [A, B, ...]} (using a list to handle possible issue with plural pronouns
    '''
    
    for token in row['tokens']:
        
        # if token is pronoun, add random character name to token
        if token['pos'] == 'PRON':
            token['char'] = [np.random.choice(charList)]

    return row['tokens']


### Model 1: Current and adjacent speakers (2 speaker model)
def pronResolution_nn(charList, row):
    '''
    I => current speaker, you => previous or next speaker
    '''
    
    for token in row['tokens']:
        
        # if token is pronoun, add character name to token
        if token['pos'] == 'PRON':
            type = token['content']
            
            # if token is "I",
            if type.lower() == 'i':
                token['char'] = row['speaker']
                
            # else, if token is "you", add previous or next speaker to dialogue
            elif type.lower() == 'you':
                token['char'] = np.random.choice([row['speaker_prev'], row['speaker_next']])
                
            # else, add random character name to token
            else:
                token['char'] = [np.random.choice(charList)]

    return row['tokens']

### Model 1.1: Current and adjacent speakers (2 speaker model)
def pronResolution_nnMod(charList, row):
    '''
    I => current speaker, you => previous or next speaker
    '''
    
    for token in row['tokens']:
        
        # if token is pronoun, add character name to token
        if token['pos'] == 'PRON':
            pLemma = token['lemma']
            
            # if token is "I",
            if pLemma.lower() in ['i', 'me']:
                token['char'] = [row['speaker']]
                
            # else, if token is "you", add previous or next speaker to dialogue
            elif pLemma.lower() == 'you':
                #get mid point and previous/next speakers
                midpoint = len(row['nearbyChars']) // 2
                prev_speaker = row['nearbyChars'][midpoint - 1]
                next_speaker = row['nearbyChars'][midpoint + 1]
                
                #count how often the current speaker appears in dialogues before and after
                #this can help with scene switches
                prev_match = sum([x == row['speaker'] for x in row['nearbyChars'][:midpoint]])
                next_match = sum([x == row['speaker'] for x in row['nearbyChars'][midpoint+1:]])
                
                #compute probability and normalize
                p = [0.5+prev_match/midpoint, 0.5+next_match/midpoint]
                p = [x / sum(p) for x in p]
                
                if not prev_speaker:
                    p = [0,1]
                if not next_speaker:
                    p = [1,0]
                
                #assign previous or next speaker based on the probability
                token['char'] = [np.random.choice([prev_speaker, next_speaker], p=p)]
                
            # else, add random character name to token
            elif pLemma.lower() not in ['what', 'it', 'this', 'that', 'those']:
                token['char'] = [np.random.choice(charList)]

    return row['tokens']


### Model 2: Speaker n-gram model:
def pronResolution_ngram():
    pass


### Model 3: Based on transition probabilities between characters
def pronResolution_hmm():
    pass


def pronResolution_sent(charDict, rows):
    '''
    cDict is a dictionary of characters and their total sentiment values in the movie
    '''
    pass



### Evaluate models
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
    pronIndex = list(df[df.num_pron > 0].index)
    
    # sample random line to evaluate resolved pronoun
    selectLine = np.random.choice(pronIndex, min(len(pronIndex), numExamples), replace=False)
    
    # for each line
    for lineNum in selectLine:
    
    # for each model df, select line to analyze
        for m in range(numModels):
            
            # select model results
            df = dfList[m]
            charList = [(x['content'], x['char']) for x in df.loc[lineNum]['tokens'] if 'char' in x]

            # print line being analyzed
            print('\n' + '*'*8 + ' line {} '.format(lineNum) + '*'*8)
            for rowNum in range(max(0, lineNum - 2), min(len(dfList[m]), lineNum + 3)):
                if rowNum == lineNum:
                    print('=> {}. {}:\n=> {}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
                else:
                    print('{}. {}:\n{}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))

            # print resolved pronouns from model
            print('*'*8 + ' test model {}: line {} '.format(m+1, lineNum) + '*'*8)
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