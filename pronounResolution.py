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
            token['char'] = np.random.choice(charList)

    return row['tokens'], row['entities']


### Model 1: Current and adjacent speakers (2 speaker model)
def pronResolution_nn(charList, row):
    '''
    I => current speaker, you => previous or next speaker
    '''
    
    for token in row['tokens']:
        
        # if token is pronoun, add character name to token
        if token['pos'] == 'PRON':
            pLemma = token['content']
            
            # if token is "I",
            if pLemma.lower() in ['i', 'me', 'my', 'mine']:
                token['char'] = [row['speaker']]
                
            # else, if token is "you", add previous or next speaker to dialogue
            elif pLemma.lower() in ['you', 'your', 'yours']:
                token['char'] = [np.random.choice([row['speaker_prev'], row['speaker_next']])]
                
            # else, add random character name to token
            else:
                token['char'] = [np.random.choice(charList)]

    return row['tokens'], row['entities']

### Model 1.1: Current and adjacent speakers (2 speaker model)
def pronResolution_nnMod(charCounter, row):
    '''
    I => current speaker, you => previous or next speaker
    '''
    
    
    for entity in row['entities']:
        
        #only run checks for person entity, change work of art to person because of marvel dataset
        if entity['type'] == 'WORK_OF_ART':
            entity['type'] = 'PERSON'
            
        if entity['type'] != 'PERSON':
            continue
            
        #first check for exact match, if so do nothing
        if entity['name'] in charCounter.keys():
            continue
        #then check for case mistmatch
        for char in charCounter.keys():
            if char.lower() == entity['name'].lower():
                entity['name'] = char
                match=True
                break
                                
        #lastly check for a partial match
        else:            
            for char in charCounter.keys():
                if entity['name'].lower() in char.lower():
                    entity['name'] = char
                    #print('found partial match')
                    break

        
    for token in row['tokens']:
        
        # if token is pronoun, add character name to token
        if token['pos'] == 'PRON':
            pLemma = token['lemma']
            
            # if token is "I",
            if pLemma.lower() in ['i', 'me', 'my', 'mine']:
                token['char'] = [row['speaker']]
                
                for entity in row['entities']:
                    if token['char'] == entity['name']:
                        entity['mentions'].append(token['content'])
                        break
                else:
                    row['entities'].append({'mentions':[token['char'][0]], 'type':'PERSON', 'name':token['char'][0]})
                
            # else, if token is "you", add previous or next speaker to dialogue
            elif pLemma.lower() in ['you', 'your', 'yours']:
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
                
                for entity in row['entities']:
                    if token['char'] == entity['name']:
                        entity['mentions'].append(token['content'])
                        break
<<<<<<< HEAD
                    else:
                        row['entities'].append({'mention':token['char'][0], 'type':'PERSON', 'name':token['char'][0]})
=======
                else:
                    row['entities'].append({'mentions':[token['char'][0]], 'type':'PERSON', 'name':token['char'][0]})
>>>>>>> cb74ee07d36a006cd1adcf45a2cb9c2a7ff40243
                
            # else, assume third person
            elif pLemma.lower() in ['he', 'his', 'him', 'her', 'hers']:
            #elif pLemma.lower() not in ['what', 'it', 'this', 'that', 'those', 'whose', 'who', 'whom', 'these',
                                       #'whosoever', 'whatever']:
                charSample = [x for x in charCounter if x not in ['narrator', row['speaker']]]
                charSum = sum([charCounter[x] for x in charCounter if x not in ['narrator', row['speaker']]])
                pSample = [charCounter[x]/charSum for x in charCounter if x not in ['narrator', row['speaker']]]
                
                token['char'] = [np.random.choice(charSample, p=pSample)]
                
                for entity in row['entities']:
                    if token['char'] == entity['name']:
                        entity['mentions'].append(token['content'])
                        break
                    else:
                        row['entities'].append({'mention':token['char'][0], 'type':'PERSON', 'name':token['char'][0]})
                   

    return row['tokens'], row['entities']



### Evaluate models
def pronEval(scripts):
    '''
    This function takes a list of dfs and evaluate model performance
    Enhance the original google api df with one of the model functions
    '''
    numScripts = len(scripts)
    sampled = np.zeros(numScripts)
    correct = np.zeros(numScripts)
    
    # for each model df, select line to analyze
    for i, scriptNum in enumerate(scripts.keys()):
        
        # select model results for script
        script = scripts[scriptNum]
        df = script['df']
        evalLines = script['eval']
        # print (i, scriptNum, evalLines)
        
        # for each line to evaluate
        for lineNum in evalLines:
            # resulting list of pronouns and referenced characters
            charList = [(x['content'], x['char']) for x in df.loc[lineNum]['tokens'] if 'char' in x]

            # print main line being analyzed, 2 lines before/after
            print('\n' + '*'*8 + ' line {} '.format(lineNum) + '*'*8)
            for rowNum in range(max(0, lineNum - 2), min(len(df), lineNum + 3)):
                speaker = df.loc[rowNum]['speaker']
                dialogue = df.loc[rowNum]['dialogue']
                
                if rowNum == lineNum:
                    print('=> {}. {}:\n=> {}\n'.format(rowNum, speaker, dialogue ))
                else:
                    print('{}. {}:\n{}\n'.format(rowNum, speaker, dialogue))

            # print resolved pronouns from model
            print('*'*8 + ' evaluate line {} in {} '.format(lineNum, script['name']) + '*'*8)
            print('{} pronouns resolved'.format(len(charList)))
            for j, char in enumerate(charList):
                print('%i. %s => %s' % (j+1, char[0].encode('utf-8'), char[1].encode('utf-8')))
                      
            collectInput = False
            
            # prompt user for count of correctly resolved pronouns
            while not collectInput:
                try:
                    count = int(input('\nhow many are correctly identified? '))
                    collectInput = True
                except:
                    print('incorrect input, only numbers allowed')

            # update counts of total/correct examples
            sampled[i] += len(charList)
            correct[i] += count

            df.set_value(lineNum, 'correct', count)
    
    # calculate and print precision for all scripts
    print('\n' + '*'*8 + ' test results ' + '*'*8)
    for i, modelResult in enumerate(zip(correct, sampled)):
        if modelResult[0] == 0:
            result = 0
        else:
            result = modelResult[0]/modelResult[1]
        print('script %i: precision = %.2f (%i/%i correct)'%(i+1, result, correct[i], sampled[i]))
    result = sum(correct)/sum(sampled)
    print('overall: precision = %.2f (%i/%i correct)'%(result, sum(correct), sum(sampled)))