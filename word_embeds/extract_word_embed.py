import pickle
from typing import Dict, List
import os
from os.path import dirname, abspath
import csv

import gensim
import gensim.downloader as api
import numpy as np

CUR_DIR = dirname(abspath(__file__))
NER_FILE = os.path.join(CUR_DIR, 'CrossNER_NOUN_PRON.pickle')
OUT_FILE = os.path.join(CUR_DIR, 'embeddings.txt')

CONCEPTNET_EMBED = 'conceptnet-numberbatch-17-06-300'
PREFIX = '/c/en/'

word_embeddings = api.load(CONCEPTNET_EMBED)
word_matrix = np.array(word_embeddings.wv.syn0)
avg_word_vec = np.mean(word_matrix, axis=0)


def get_word_embed(word: str) -> np.ndarray:
    '''
    Gets the word embeddings for a given word from Conceptnet in a safe manner.
    Just calling `word_embeddings[word]` may fail, so a number of fallbacks are
    implemented.
    '''
    clean_word = word.lower().replace(' ', '_')
    p_word = PREFIX + clean_word

    # if we have the word, return it directly
    if p_word in word_embeddings:
        return word_embeddings[p_word]

    # Some words like 'autoencoders' are not present, but 'autoencoder' is.
    # Other words like '-binding' is passed in, but only 'binding' exists
    # It is likely that the shorter word carries more meaning than not having
    # anything at all
    for i in range(1, len(clean_word)):
        # Truncating the suffix by i characters
        word_truncated_suffix = p_word[:-i]
        if word_truncated_suffix in word_embeddings:
            return word_embeddings[word_truncated_suffix]

        # Truncating the prefix by i characters
        word_truncated_prefix = PREFIX + clean_word[i:]
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
                word = word_tuple[0]
                all_words[word] = get_word_embed(word)
    return all_words


def to_textfile_line(word: str, vec: np.ndarray) -> str:
    '''
    Takes a word and its corresponding vector and returns a string
    representing the output to the text file
    '''
    return f'{word} {" ".join([str(num) for num in vec])}\n'


def read_embeddings(embedding_file: str = OUT_FILE) -> Dict[str, np.ndarray]:
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
    with open(OUT_FILE, 'a') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
