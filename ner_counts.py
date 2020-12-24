#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
output Stanford NER results (including entity counts and entity/tag frequency rankings)

Usage:
1. Download Stanford NLP's NER package at 
      [https://nlp.stanford.edu/software/CRF-NER.html#Download].
2. Take a look at the directory paths in process(),
   and set up the script accordingly.
2. Make sure 
       stanford-ner.jar
   and 
       english.all.3class.distsim.crf.ser.gz
   are stored in the correct place.
4. Execute the script: 
        python ner_counts.py metadata.csv FEM|WOMEN|SAMPLE
"""
import sys
import time
import pathlib
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tag.stanford import StanfordNERTagger


# In[ ]:


def create_all_file_list(path=''):
    global CORPUS_PATH
    
    if not path:
        temp = []
        for hathi_id in hathi_ids:
            temp.append(sorted(pathlib.Path(CORPUS_PATH).glob(hathi_id + '/*.txt')))
        return [str(corpus_file) for corpus_files in temp for corpus_file in corpus_files]
    else:
        return [str(corpus_file) 
                for corpus_file in sorted(list(path.glob('*.txt')))]

def process(label):
    global CORPUS_PATH, LIB_PATH, OUTPUT_PATH
    
    NER_CORPUS = CORPUS_PATH.joinpath(label)

    files = create_all_file_list(path=NER_CORPUS)

    for file in files:
        output_filename = file.split('/')[-1].split('.txt')[0]
        with open(file, 'r') as f:
            sentence = f.read()

        ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
        words = word_tokenize(sentence)
        entities_list = ner_tagger.tag(words)

        entities_dict = {}
        for entity, tag in entities_list:
            if (tag != 'O') and (tag not in entities_dict):
                entities_dict[tag] = []
            if tag != 'O':
                entities_dict[tag].append(entity)

        entities_df = pd.DataFrame(list(entities_dict.items()),
                                   columns=['tag', 'entities']).set_index('tag')


        csv_filename = str(OUTPUT_PATH.joinpath(output_filename + '_entities_content.csv'))
        entities_df.to_csv(csv_filename)
        print('Entities info saved to {}.'.format(csv_filename))

        ranking_list = [entity for subl in entities_dict.values() for entity in subl]
        entity_counts = Counter(ranking_list)
        entities_ranking = pd.DataFrame.from_dict(entity_counts, orient='index').rename(columns={0: 'freq'})
        entities_ranking.sort_values(by=['freq'], ascending=False, inplace=True)

        csv_filename = str(OUTPUT_PATH.joinpath(output_filename + '_entities_ranking.csv'))
        entities_ranking.to_csv(csv_filename)
        print('Entities ranking saved to {}.'.format(csv_filename))    

        _, tag_list = zip(*entities_list)
        tag_counts = Counter(tag_list)
        tag_ranking = pd.DataFrame.from_dict(tag_counts, orient='index').rename(columns={0: 'freq'})

        csv_filename = str(OUTPUT_PATH.joinpath(output_filename + '_tag_ranking.csv'))
        tag_ranking.to_csv(csv_filename)
        print('Tag ranking saved to {}.'.format(csv_filename))
        
if __name__ == "__main__":
    metadata_filename = 'metadata.csv'#sys.argv[1]
    label = 'FEM' #sys.argv[2]
    
    SCRIPT_PATH = pathlib.Path.cwd()
    METADATA_PATH = SCRIPT_PATH.joinpath(metadata_filename)
    
    metadata_df = pd.read_csv(METADATA_PATH)
    labels = list(set(metadata_df['CLASS'].to_list()))
    
    if label not in labels:
        print('Error: label not found; should be one of the following: '              +'|'.join(labels))
        raise Exception
    
    # make sure the paths are correct
    jar = './LIB/stanford-ner/stanford-ner.jar'
    model = './LIB/stanford-ner/english.all.3class.distsim.crf.ser.gz'

    RAW_PATH = SCRIPT_PATH.joinpath('TEXTS')
    CORPUS_PATH = SCRIPT_PATH.joinpath('CORPUS')
    OUTPUT_PATH = SCRIPT_PATH.joinpath('OUTPUT_ENTITIES')

    process(label)


# In[ ]:





# In[ ]:




