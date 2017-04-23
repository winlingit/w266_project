import re
import nltk
import numpy as np
import pandas as pd
import os
import json

def getRelations():
    return {0:'others', 1:'negative mentioning', 2:'positive mentioning', 3:'group mentioning', 4: 'mixed mentioning', 5: 'place mentioning'}

def simpleRE(rows):
    '''
    takes in tokens and extract potential relations
    if 'char' is available for a token, use the value instead of the actual content
    '''
    relation = []
    nsubj = -1
    verb = -1
    print 'rows.token: ' + rows.token
    for i, token in enumerate(rows.tokens):
        print 'i: ' + i
        print 'token: ' + token
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
    
def extract_relation_categories(rows):
    relation = []
    team_relations = extract_mention_team(rows)
    places_mentioned = extract_place_mentioned(rows)
    mention_sentiment_relations = extract_mention_sentiment(rows)
    if team_relations:
        relation.extend(team_relations)
    if places_mentioned:
        relation.extend(places_mentioned)
    if mention_sentiment_relations:
        relation.extend(mention_sentiment_relations)
    
    if relation:
        return relation
    else:
        return None    

def extract_mention_team(rows):
    relation = []
    num_other_persons_mentioned = len([e for e in rows.entities if e['type'] == 'PERSON' and e['name'] not in rows.speaker])
    if num_other_persons_mentioned > 2:
        persons_list = [e for e in rows.entities if e['type'] == 'PERSON' and e['name'] not in rows.speaker]
        ent2 = ''
        for e in persons_list:
            ent2 += ', ' + e['name']
        relation.append({'relation':rows.dialogue, 
                             'ent1':rows.speaker, 'ent2':ent2, 'class':3, 'line':rows.name})
        return relation
    else:
        return None
        
def extract_place_mentioned(rows):
    relation = []
    places_list = [e for e in rows.entities if e['type'] == 'LOCATION']
    
    for p in places_list:
        rel_index = -1
        place_is_subj = False
        
        for i, token in enumerate(rows.tokens):
            rel_phrase = ''
            if token['label'] == 'NSUBJ' and token['content'] in p['name'].split(' '):
                nsubj = i
                rel_index = token['index']
                place_is_subj = True
                
            if token['label'] == 'NSUBJ':
                nsubj = i
                rel_index = token['index']
                
            if 'OBJ' in token['label'] and token['index'] == rel_index and token['content'] in p['name'].split(' '):
                subj = rows.tokens[nsubj].get('char', rows.tokens[nsubj]['content'])
                obj = p['name']
                
                for j in range(rel_index, i):
                    rel_phrase += ' ' + rows.tokens[j]['content']
                relation.append({'relation': rel_phrase, 
                                 'ent1':subj, 'ent2':obj, 'class':5, 'line':rows.name})
                
            if 'OBJ' in token['label'] and token['index'] == rel_index and place_is_subj:
                subj = p['name']
                obj = rows.tokens[i].get('char', rows.tokens[i]['content'])
                
                for j in range(rel_index, i):
                    rel_phrase += ' ' + rows.tokens[j]['content']
                relation.append({'relation': rel_phrase, 
                                 'ent1':subj, 'ent2':obj, 'class':5, 'line':rows.name})
        
        if relation:
            return relation
        else:
            return None
    

def extract_mention_sentiment(rows):
    
    relation = []
    persons_list = [e for e in rows.entities if e['type'] == 'PERSON' and e['name'] not in rows.speaker]
    sentiment_score = rows.sentiment['score']
    sentiment_mag = rows.sentiment['magnitude']
    ent2 = ''
    
    for e in persons_list:
        ent2 += e['name']
        
    if not persons_list:
        return None
    
    elif abs(sentiment_score) > 0.5:
        for p in persons_list:
            relation.append({'relation': rows.dialogue, 
                             'ent1':rows.speaker, 'ent2': p['name'], 'class': (sentiment_score > 0) + 1, 'line':rows.name})
                         
    elif rows.sentiment['magnitude'] > 2:
        relation.append({'relation': rows.dialogue, 
                             'ent1':rows.speaker, 'ent2': ent2, 'class': 4, 'line':rows.name})
        
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
            print('\n' + '*'*8 + ' line {} '.format(lineNum) + '*'*8)
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
