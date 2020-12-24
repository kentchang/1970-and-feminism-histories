#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
create necessary directories and hathi_ids.txt
"""
import sys
import pathlib
import pandas as pd

def process(metadata_filename):
    SCRIPT_PATH = pathlib.Path.cwd()

    RAW_PATH = SCRIPT_PATH.joinpath('TEXTS')
    CORPUS_PATH = SCRIPT_PATH.joinpath('CORPUS')
    LIB_PATH = SCRIPT_PATH.joinpath('LIB')
    COLLOCATION_PATH = SCRIPT_PATH.joinpath('OUTPUT_COLLOCATIONS')
    ENTITY_PATH = SCRIPT_PATH.joinpath('OUTPUT_ENTITIES')
    TEMP_PATH = COLLOCATION_PATH.joinpath('TEMP')
    
    if not RAW_PATH.is_dir():
        RAW_PATH.mkdir()
    if not CORPUS_PATH.is_dir():
        CORPUS_PATH.mkdir()
    if not LIB_PATH.is_dir():
        LIB_PATH.mkdir()    
    if not COLLOCATION_PATH.is_dir():
        COLLOCATION_PATH.mkdir()   
    if not ENTITY_PATH.is_dir():
        ENTITY_PATH.mkdir()           
    if not TEMP_PATH.is_dir():
        TEMP_PATH.mkdir()

    print('Script path: {}'.format(str(SCRIPT_PATH)))
    print('Hathi raw text path: {}'.format(str(RAW_PATH)))
    print('Labeled corpora to be stored in: {}'.format(str(CORPUS_PATH)))
    print('External libraries should go to: {}'.format(str(LIB_PATH)))
    print('Collocation stats will be saved in: {}'.format(str(COLLOCATION_PATH)))
    print('Named entity counts will be stored in: {}'.format(str(ENTITY_PATH)))
    print('Temporary folder is: {}'.format(str(TEMP_PATH)))
    
    METADATA_PATH = SCRIPT_PATH.joinpath(metadata_filename)
    metadata_df = pd.read_csv(METADATA_PATH)
    labels = list(set(metadata_df['CLASS'].to_list()))
    print('Available labels: {}'.format('|'.join(labels)))
    
    hathi_ids = list(set(metadata_df['HATHI_ID'].to_list()))
    with open('hathi_ids.txt', 'w+') as f:
        f.write('\n'.join(hathi_ids))
    print('Saved Hathi ID file `hathi_ids.txt`.')
    print('Now `htrc download -c -o $PWD/CORPUS hathi_ids.txt` in your capsule under secure mode.')


# In[ ]:


if __name__ == "__main__":
    process(sys.argv[1])


# In[ ]:




