import pickle
import csv

from sklearn.metrics import v_measure_score, adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import pandas as pd

GLOVE = 'GloVe'
CONCEPTNET = 'Conceptnet'
WORD2VEC = 'Word2Vec'
FAST_TEXT = 'FastText'

with open('CrossNER_NOUN_PRON.pickle', 'rb') as f:
    NER_DICT = pickle.load(f)


politics = ['politician', 'political party', 'election']
nat_science = [
    'scientist', 'university', 'discipline', 'enzyme', 'protein',
    'chemical compound', 'chemical element', 'astronomical object',
    'academic journal', 'theory'
]
music = [
    'music genre', 'song', 'band', 'album', 'musical artist',
    'musical instrument'
]
lit = ['book', 'writer', 'poem', 'event', 'magazine']
ai = [
    'field', 'task', 'product', 'algorithm', 'researcher', 'metrics',
    'university'
]

domains = [politics, nat_science, music, lit, ai]


labels = {
    'Politics': politics,
    'Natural Science': nat_science,
    'Music': music,
    'Literature': lit,
    'AI': ai,
}

label_nums = {
    'Politics': 0,
    'Natural Science': 1,
    'Music': 2,
    'Literature': 3,
    'AI': 4
}

NUM_DOMAINS = len(labels)


def read_embeddings(embedding_file: str):
    words_df = pd.read_csv(embedding_file, sep=" ", index_col=0,
                           na_values=None, keep_default_na=False,
                           header=None, quoting=csv.QUOTE_NONE)
    word_embeddings = {word: words_df.loc[word].values
                       for word in words_df.index.values}
    return word_embeddings


def transform_data(word_embeddings):
    X, Y = [], []

    for label, words in labels.items():
        for word in words:
            Y.append(label_nums[label])
            clean_word = word.replace(' ', '_')
            if clean_word in word_embeddings:
                word_vec = word_embeddings[clean_word]
            else:
                word_vec = np.mean([word_embeddings[subword] for subword
                                    in word.split()], axis=0)
            X.append(word_vec)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def transform_data_ner(word_embeddings):
    X, Y = [], []

    for label_num, domain in enumerate(NER_DICT.keys()):
        for pos in NER_DICT[domain].keys():
            for word_info in NER_DICT[domain][pos]:
                word = word_info[0]
                clean_word = word.replace(' ', '_')
                if clean_word in word_embeddings:
                    word_vec = word_embeddings[clean_word]
                else:
                    word_vec = np.mean([word_embeddings[subword] for subword
                                        in word.split()], axis=0)

                X.append(word_vec)
                Y.append(label_num)

    X, Y = np.array(X), np.array(Y)
    return X, Y


def score_word_embeds(X, Y):
    kmeans = KMeans(n_clusters=NUM_DOMAINS, random_state=0).fit(X)
    rand = adjusted_rand_score(Y, kmeans.labels_)
    v = v_measure_score(Y, kmeans.labels_)
    return rand, v


def main():
    rows = []
    # Add FastText as well if you have the embeds
    for embed in tqdm([GLOVE, CONCEPTNET]):
        word_embeddings = read_embeddings(f'embeddings-{embed.lower()}.txt')
        for desc, transform_func in [('Labels', transform_data),
                                     ('Concepts', transform_data_ner)]:
            X, Y = transform_func(word_embeddings)
            rand, v_measure = score_word_embeds(X, Y)
            rows.append([desc, embed, rand, v_measure])

    cols = ['Description', 'Word Embed', 'Adjusted Rand Score', 'V Measure']
    print(pd.DataFrame(rows, columns=cols).to_string())


if __name__ == '__main__':
    main()
