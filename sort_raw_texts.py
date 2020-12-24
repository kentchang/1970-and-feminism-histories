#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Produce named entity counts 
Usage: python sort_raw_texts.py metadata.csv
"""

import sys
import pathlib
import pandas as pd
import random


# In[6]:


def create_corpus_file_list(class_label, from_raw=False):
    global RAW_PATH, CORPUS_PATH, METADATA_PATH

    class_label = class_label.upper()
    metadata_df = pd.read_csv(METADATA_PATH)
    hathi_ids = metadata_df.loc[metadata_df['CLASS'] == class_label]["HATHI_ID"].to_list()
    
    temp = []
    if from_raw:
        for hathi_id in hathi_ids:
            temp.append(str(RAW_PATH.joinpath(hathi_id + '.txt')))
        return sorted(temp)
    else:
        for hathi_id in hathi_ids:
            temp.append(sorted(pathlib.Path(CORPUS_PATH).glob(hathi_id + '/*.txt')))
        return [str(corpus_file) for corpus_files in temp for corpus_file in corpus_files]

    
def process(metadata_filename):
    global RAW_PATH, CORPUS_PATH, METADATA_PATH

    metadata_df = pd.read_csv(METADATA_PATH)
    labels = list(set(metadata_df['CLASS'].to_list()))
    commands = []
    for label in labels:
        if label != 'SAMPLE': 
            corpus_file_list = create_corpus_file_list(label, True)
            if not CORPUS_PATH.joinpath(label).is_dir():
                CORPUS_PATH.joinpath(label).mkdir() 
        else:
            all_sample = create_corpus_file_list('SAMPLE', True)
            corpus_file_list = random.sample(all_sample, 300)
        for corpus_file in corpus_file_list:
            filename = corpus_file.split('/')[-1]
            commands.append('cp ' + "'" + corpus_file + "'" +                             ' ' +                             "'" + str(CORPUS_PATH.joinpath('{}/{}'.format(label, filename))) + "'"                           )

    with open('SORT_RAW_TEXTS.sh', 'w+') as f:
        f.write('\n'.join(commands))
        print("Created SORT_RAW_TEXTS.sh") 
        print("Issue `chmod +x SORT_RAW_TEXTS.sh` and execute the bash script.")


# In[7]:


if __name__ == "__main__":
    metadata_filename = 'metadata.csv'#sys.argv[1]
    SCRIPT_PATH = pathlib.Path.cwd()
    METADATA_PATH = SCRIPT_PATH.joinpath(metadata_filename)
    RAW_PATH = SCRIPT_PATH.joinpath('TEXTS')
    CORPUS_PATH = SCRIPT_PATH.joinpath('CORPUS')
    if not CORPUS_PATH.is_dir():
        CORPUS_PATH.mkdir()
    METADATA_PATH = SCRIPT_PATH.joinpath(metadata_filename)
    
    process(metadata_filename)


# In[ ]:




