#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Generate AntConc-like stats for collocations.

Usage: python collocation_stats_viewer METADATA_FILENAME CLASS_LABEL LEFT_BOUNDARY RIGHT_BOUNDARY FREQUENCY_THRESHOLD IS_AGGREGATE TERMS_FILE
"""
import sys
import re
import copy
import csv
import pathlib
import math
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.text import ConcordanceIndex
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import regexp_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramAssocMeasures, QuadgramCollocationFinder
from nltk.tokenize import RegexpTokenizer


# In[ ]:


def fast_tokenize(txt_as_string):
    txt_as_string = txt_as_string.lower()
    tokens = re.findall(r'(\w+)', txt_as_string, re.UNICODE)
    tokens = [token 
              for token in tokens 
              if not set('_0123456789')&set(token)]
    return tokens

def concordance(ci, word, width=60):
    global boundary
    
    half_width = (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context
    right_collocates_list = []
    left_collocates_list = []
    
    offsets = ci.offsets(word)
    if offsets:
        for i in offsets:
            left_context = (' '.join(ci._tokens[i-context:i]))
            right_context = ' '.join(ci._tokens[i+1:i+context])
            left_collocates = ci._tokens[i-boundary[0]:i]
            right_collocates = ci._tokens[i+1:i+boundary[1]+1]
            if len(right_context[:half_width])==0:
                left_collocates_list.append(left_collocates)
            elif len(left_context[-half_width:])==0:
                right_collocates_list.append(right_collocates)
            else:
                left_collocates_list.append(left_collocates)
                right_collocates_list.append(right_collocates)
    return left_collocates_list, right_collocates_list

def mi(interested_term, filtered_collocate, word_fd, filtered_collocate_fd):
    f_nc = filtered_collocate_fd[filtered_collocate]
    f_n = word_fd[interested_term]
    f_c = word_fd[filtered_collocate]
    N = sum(word_fd.values())
    O_over_E = (f_nc * N) / (f_n * f_c)
    return 0 if O_over_E==0 else round(math.log(O_over_E,2),5)

def save_output(output_list):
    global OUTPUT_PATH, TEMP_PATH, filtered_collocate_fd, interested_term
    
    tsv_filename = 'collocationstats_' + label + '_' + interested_term + '.tsv'
    tsv_strpath = str(OUTPUT_PATH.joinpath(tsv_filename))
    
    if (len(output_list)==1):
        print(output_list)
        with open(tsv_strpath, 'w+') as o:
            o.write(''.join(output_list))
            print('Error! Msg saved to {}.'.format(tsv_filename))
    else:
        filtered_collocates = list(filtered_collocate_fd.keys())
        filtered_collocates_type_count = len(filtered_collocates)
        filtered_collocates_token_count = sum(list(filtered_collocate_fd.values()))

        df = pd.DataFrame(output_list).sort_values(by=['Stat'], ascending=False)
        df = df.reset_index(drop=True)
        df.index = df.index+1
        df.index.name = 'Rank'

        temp_tsv_filename = 'temp_' + 'collocationstats_' + label + '_' + interested_term + '.tsv'
        temp_tsv_strpath = str(TEMP_PATH.joinpath(temp_tsv_filename))

        df.to_csv(temp_tsv_strpath, sep='\t', quoting=csv.QUOTE_NONE)

        with open(temp_tsv_strpath, 'r') as f:
            contents = f.read()
            with open(tsv_strpath, 'w+') as o:
                o.write('#Collocation stats for: {}\n'.format(interested_term))
                o.write('#Total No. of Collocate Types: {}\n'.format(str(filtered_collocates_type_count)))
                o.write('#Total No. of Collocate Tokens: {}\n'.format(str(filtered_collocates_token_count)))
                o.write(contents)
                print('Collocation info saved to {}.'.format(tsv_filename))

def lookup_interested_term(interested_term, word_fd, filtered_collocate_fd, collocate_fd, left_collocate_fd, right_collocate_fd):
    global freq_threshold
    
    if interested_term not in word_fd.keys():
        print('{} not in tokens!'.format(interested_term))
        return ['{} not in tokens!'.format(interested_term)]

    filtered_collocates = list(filtered_collocate_fd.keys())
    
    if not filtered_collocates:
        print('No word that collocates with {} has a frequency over {}!'.format(interested_term, freq_threshold))
        return ['No word that collocates with {} has a frequency over {}!'.format(interested_term, freq_threshold)]
        
    result_list = []
    for filtered_collocate in filtered_collocates:
        result_entry = {
            'Freq': collocate_fd[filtered_collocate],
            'Freq_(L)': left_collocate_fd[filtered_collocate], 
            'Freq_(R)': right_collocate_fd[filtered_collocate], 
            'Stat': mi(interested_term, filtered_collocate, word_fd, filtered_collocate_fd),
            'Collocate': filtered_collocate
        }
        result_list.append(result_entry)
    
    return result_list

def token_to_fds(tokens, interested_term):
    global freq_threshold, aggregate_results

    ci = ConcordanceIndex(tokens)
    left_collocates_list, right_collocates_list = concordance(ci, interested_term)

    left_collocates_raw = [left_collocate for sublist in left_collocates_list for left_collocate in sublist]
    right_collocates_raw = [right_collocate for sublist in right_collocates_list for right_collocate in sublist]
    left_collocate_fd = FreqDist(left_collocates_raw)
    right_collocate_fd = FreqDist(right_collocates_raw)
    collocate_fd = copy.deepcopy(right_collocate_fd)
    collocate_fd.update(left_collocate_fd)
    
    if not aggregate_results:
        # i.e. the collocate_fd above is final; if we do need to aggregate results, we will have to pen other files, so c_fd is not yet final and shouldn't produce filtered fd
        filtered_collocate_fd = {
            collocate: freq
            for collocate, freq in collocate_fd.items()
            if freq >= freq_threshold
        }
        return left_collocate_fd, right_collocate_fd, collocate_fd, filtered_collocate_fd
    return left_collocate_fd, right_collocate_fd, collocate_fd


# In[ ]:





# In[ ]:


def process(class_label, freq_threshold, boundary, is_aggregate, interested_terms_filename):
    CLASS_LABEL = class_label
    corpus_files = create_corpus_file_list(CLASS_LABEL, True)
    aggregate_results = is_aggregate
    with open(interested_terms_filename, 'r') as f:
        interested_terms = f.read().rstrip().split('\n')

    if aggregate_results:
        label = CLASS_LABEL

        for interested_term in interested_terms:
            print('term: '+interested_term)
            word_fd = FreqDist()
            left_collocate_fd = FreqDist()
            right_collocate_fd = FreqDist()
            collocate_fd = FreqDist()
            filtered_collocate_fd = FreqDist()
            for corpus_file in corpus_files:
                with open(str(corpus_file), 'r',
                      encoding='utf8', errors='ignore') as file:
                    text = file.read()
                    tokens = fast_tokenize(text)
                    tokens = list(filter(None, tokens))
                new_fds = token_to_fds(tokens, interested_term)
                word_fd.update(FreqDist(tokens))
                left_collocate_fd.update(new_fds[0]) 
                right_collocate_fd.update(new_fds[1])
                collocate_fd.update(new_fds[2])
            filtered_collocate_fd.update({
                collocate: freq
                for collocate, freq in collocate_fd.items()
                if int(freq) >= int(freq_threshold)
            })
            output_list = lookup_interested_term(interested_term,
                                                 word_fd,
                                                 filtered_collocate_fd, 
                                                 collocate_fd, 
                                                 left_collocate_fd, 
                                                 right_collocate_fd)
            if output_list:
                save_output(output_list)
    else:
        for corpus_file in corpus_files:
            label = corpus_file.split('/')[-1].split('.txt')[0]
            with open(str(corpus_file), 'r',
                  encoding='utf8', errors='ignore') as file:
                text = file.read()
                tokens = fast_tokenize(text)
                tokens = list(filter(None, tokens))
                word_fd = FreqDist(tokens)
                print('Word Types: {}'.format(str(len(word_fd))))
                print('Word Tokens: {}'.format(str(sum(word_fd.values()))))
                for interested_term in interested_terms:
                    print('term: '+interested_term)
                    left_collocate_fd,                    right_collocate_fd, collocate_fd,                    filtered_collocate_fd =                        token_to_fds(tokens, interested_term)
                    output_list = lookup_interested_term(interested_term,
                                                         word_fd,
                                                         filtered_collocate_fd, 
                                                         collocate_fd, 
                                                         left_collocate_fd, 
                                                         right_collocate_fd)
                    if output_list:
                        save_output(output_list)


# In[ ]:


if __name__ == "__main__":
    metadata_filename = sys.argv[1]
    class_label = sys.argv[2]
    boundary = (sys.argv[3], sys.argv[4])
    freq_threshold = sys.argv[5]
    is_aggregate = sys.argv[6]
    interested_terms_filename = sys.argv[7]

    SCRIPT_PATH = pathlib.Path.cwd()
    RAW_PATH = SCRIPT_PATH.joinpath('TEXTS')
    CORPUS_PATH = SCRIPT_PATH.joinpath('CORPUS')
    LIB_PATH = SCRIPT_PATH.joinpath('LIB')
    OUTPUT_PATH = SCRIPT_PATH.joinpath('OUTPUT_COLLOCATIONS')
    TEMP_PATH = OUTPUT_PATH.joinpath('TEMP')
    METADATA_PATH = SCRIPT_PATH.joinpath(metadata_filename)  
    
    process(class_label, freq_threshold, boundary, is_aggregate, interested_terms_filename)


# In[ ]:





# In[ ]:





# In[ ]:




