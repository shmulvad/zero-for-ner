import pickle
from typing import Dict, List
import os
from os.path import dirname, abspath
import csv

import gensim
import gensim.downloader as api
import pandas as pd
import numpy as np

CONCEPTNET = 'Conceptnet'
GLOVE = 'GloVe'
WORD2VEC = 'Word2Vec'
FAST_TEXT = 'FastText'

### Set this constant based on which word embedding you want to start from ###
EMBED = GLOVE


CUR_DIR = dirname(abspath(__file__))
NER_FILE = os.path.join(CUR_DIR, 'CrossNER_NOUN_PRON.pickle')

WORD_EMBED_TO_FILE = {
    GLOVE: 'glove-wiki-gigaword-300',
    CONCEPTNET: 'conceptnet-numberbatch-17-06-300',
    WORD2VEC: 'word2vec-google-news-300',
    FAST_TEXT: 'fasttext-wiki-news-subwords-300'
}

ADDITIONAL_WORDS = [
    'academic journal',
    'album',
    'algorithm',
    'astronomical object',
    'award',
    'band',
    'book',
    'chemical compound',
    'chemical element',
    'country',
    'discipline',
    'election',
    'enzyme',
    'event',
    'field',
    'location',
    'magazine',
    'metrics',
    'miscellaneous',
    'music genre',
    'musical artist',
    'musical instrument',
    'organization',
    'person',
    'poem',
    'political party',
    'politician',
    'product',
    'protein',
    'researcher',
    'scientist',
    'song',
    'task',
    'theory',
    'university',
    'writer'
]

prefix = '/c/en/' if EMBED == CONCEPTNET else ''
embedding_file = WORD_EMBED_TO_FILE[EMBED]
word_embeddings = api.load(embedding_file)
word_matrix = np.array(word_embeddings.wv.syn0)
avg_word_vec = np.mean(word_matrix, axis=0)


def get_word_embed(word):
    clean_word = word.lower()
    p_word = prefix + clean_word

    # if we have the word, return it directly
    if p_word in word_embeddings:
        return word_embeddings[p_word]

    # If multi-words token like 'political party', and 'political_party' was
    # not present in embeddings, return the mean of the individual embeddings
    if '_' in clean_word:
        subvecs = [get_word_embed(subword) for subword in word.split('_')]
        return np.mean(subvecs, axis=0)


    # Some words like 'autoencoders' are not present, but 'autoencoder' is.
    # Other words like '-binding' is passed in, but only 'binding' exists
    # It is likely that the shorter word carries more meaning than not having
    # anything at all
    for i in range(1, len(clean_word)):
        word_truncated_suffix = p_word[:-i]
        if word_truncated_suffix in word_embeddings:
            return word_embeddings[word_truncated_suffix]

        word_truncated_prefix = prefix + clean_word[i:]
        if word_truncated_prefix in word_embeddings:
            return word_embeddings[word_truncated_prefix]

    # Finally, return the average vector if nothing else.
    # Note this is different from a vector of purely zeros
    return avg_word_vec


def transform_ner(ner_dict: Dict[str, Dict[str, List[str]]]) \
                  -> Dict[str, np.ndarray]:
    '''
    Loop over all the words in the NER_dict and get their corresponding
    word embedding
    '''
    all_words = {}

    for topic in ner_dict.keys():
        for pos in ner_dict[topic].keys():
            for word_tuple in ner_dict[topic][pos]:
                word = word_tuple[0].replace(' ', '_')
                all_words[word] = get_word_embed(word)

    for raw_word in ADDITIONAL_WORDS + list(ner_dict.keys()):
        word = raw_word.replace(' ', '_')
        all_words[word] = get_word_embed(word)

    return all_words


def to_textfile_line(word: str, vec: np.ndarray) -> str:
    '''
    Takes a word and its corresponding vector and returns a string
    representing the output to the text file
    '''
    return f'{word} {" ".join([str(num) for num in vec])}\n'


def read_embeddings(embedding_file: str) -> Dict[str, np.ndarray]:
    '''Reads the embeddings and returns a dictionary of these'''
    words_df = pd.read_csv(embedding_file, sep=" ", index_col=0,
                           na_values=None, keep_default_na=False,
                           header=None, quoting=csv.QUOTE_NONE)
    embeddings_index = {word: words_df.loc[word].values
                        for word in words_df.index.values}
    return embeddings_index


def main():
    with open(NER_FILE, 'rb') as f:
        ner_dict = pickle.load(f)

    all_words = transform_ner(ner_dict)
    lines = [to_textfile_line(word, vec) for word, vec in all_words.items()]
    filename = f'embeddings-{EMBED.lower()}.txt'
    with open(os.path.join(CUR_DIR, filename), 'a') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
