import re
import nltk
import numpy as np
import pandas as pd
import os

def simpleRE(rows):
    '''
    takes in tokens and extract potential relations
    if 'char' is available for a token, use the value instead of the actual content
    '''
    relation = []
    nsubj = -1
    verb = -1
    for i, token in enumerate(rows.tokens):
        if token['label'] == 'NSUBJ':
            nsubj = i
            verb = token['index']
        if 'OBJ' in token['label'] and token['index'] == verb:
            #print(tokens[nsubj])
            subj = rows.tokens[nsubj].get('char', rows.tokens[nsubj]['content'])
            obj = rows.tokens[i].get('char', rows.tokens[i]['content'])
            relation.append({'relation':rows.tokens[verb]['content'], 
                             'ent1':subj, 'ent2':obj, 'class':0, 'line':rows.name})
        
    if relation:
        return relation
    else:
        return None
    
def REEval(dfList, numExamples=50):
    '''
    This function takes a list of dfs and evaluate model performance
    Enhance the original google api df with one of the model functions
    '''
    numModels = len(dfList)
    sampled = np.zeros(numModels)
    correct = np.zeros(numModels)
    
    # indexes for lines of dialogue with resolved pronouns
    df = dfList[0]
    REIndex = list(df[df.relations.notnull()].index)
    
    # sample lines
    selectLine = np.random.choice(REIndex, numExamples, replace=False)
    for lineNum in selectLine:

        # for each model df, select line to analyze
        for m in range(numModels):
            
            # select model results
            df = dfList[m]

            # print line being analyzed
            print('\n' + '*'*8 + ' line {} '.format(selectLine) + '*'*8)
            for rowNum in range(max(0, lineNum - 2), min(len(dfList[m]), lineNum + 3)):
                if rowNum == lineNum:
                    print('=> {}. {}:\n=> {}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
                else:
                    print('{}. {}:\n{}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))

            # print resolved pronouns from model
            print('*'*8 + ' test model {}: line {} '.format(m+1, lineNum) + '*'*8)
            print('{} relations identified'.format(len(df.loc[lineNum]['relations'])))
            for relation in df.loc[lineNum]['relations']:
                print('entities: {} => {}'.format(relation['ent1'], relation['ent2']))
                print('relation: {}'.format(relation['relation']))
                print('category: {}'.format(relation['class']))
                      

            collectInput = False

            # prompt user for count of correctly resolved pronouns
            while not collectInput:
                try:
                    count = int(input('\nhow many are correctly identified? '))
                    collectInput = True
                except:
                    print('incorrect input, only numbers allowed')

            # update counts of lines sampled from script and correctly extract relations
            sampled[m] += len(df.loc[lineNum]['relations'])
            correct[m] += count
        
    # calculate and print precision for all models
    print('\n' + '*'*8 + ' test results ' + '*'*8)
    for i, modelResult in enumerate(zip(correct, sampled)):
        if modelResult[0] == 0:
            result = 0
        else:
            result = modelResult[0]/modelResult[1]
        print('model %i: precision = %.2f (%i/%i correct)'%(i+1, result, correct[i], sampled[i]))
