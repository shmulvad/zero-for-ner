import csv
import os
from typing import List, Dict

import numpy as np
import numpy.linalg as LA
import pandas as pd
import click

EMBEDDING_FILES = ['conceptnet', 'glove', 'fasttext']

# python combine_word_embed.py --embed-files="glove,conceptnet,fasttext" \
#  --dim=300 --outfile="embeddings-combined.txt"


def read_embeddings(embedding_file: str) -> Dict[str, np.ndarray]:
    '''Reads the embeddings'''
    words_df = pd.read_csv(embedding_file, sep=" ", index_col=0,
                           na_values=None, keep_default_na=False,
                           header=None, quoting=csv.QUOTE_NONE)
    word_embeddings = {word: words_df.loc[word].values
                       for word in words_df.index.values}
    return word_embeddings


def get_vocab(*embeds) -> List[str]:
    '''Gets the vocab of the embeddings and ensure the vocab is consistent'''
    assert len(embeds) > 0, 'At least one embedding should be supplied'
    vocab = list(embeds[0].keys())

    assert all(set(vocab) == set(embed.keys()) for embed in embeds), \
        'Vocabs are different'

    return vocab


def concat_embeddings(vocab: List[str], *embeds) -> np.ndarray:
    '''
    Takes a vocab of size V and a number of embeddings each of size [V, D]
    where D is their corresponding dimensionality. Returns a matrix of size
    [V, K] where K = D1 + D2 + D3 + ...
    The kth entry in the matrix corresponds to the kth entry in vocab
    '''
    return np.array([np.hstack([embed[word] for embed in embeds])
                     for word in vocab])


def reduce_dimensionality(word_mat: np.ndarray, output_dim: int) -> np.ndarray:
    '''
    Takes a matrix of the word embeddings, word_mat, of size [V, K] where K is
    a high dimensionality and returns a matrix of size [V, output_dim] where
    as much of the variance as possible is kept

    See the following paper for details on the algorithm:
    https://arxiv.org/pdf/1704.03560.pdf
    '''
    _, s, V_T = LA.svd(word_mat)
    V_truncated = V_T.T[:, :output_dim]
    sigma_sqrt = np.diag(np.sqrt(s[:output_dim]))
    return word_mat @ V_truncated @ sigma_sqrt


def normalize(M: np.ndarray) -> np.ndarray:
    '''
    Normalizes all V vectors in the [V, output_dim] matrix M to a norm of 1
    '''
    norms_inv = 1.0 / LA.norm(M, axis=1)
    return np.einsum('i,ij->ij', norms_inv, M)


def to_textfile_line(word: str, vec: np.ndarray) -> str:
    '''
    Takes a word and its corresponding vector and returns a string
    representing the output to the text file
    '''
    nums_str = ' '.join([f'{num:.6f}' for num in vec])
    return f'{word} {nums_str}\n'


'''CODE WHERE THE OUTPUT SEEMS SKETCHY'''
# # https://openreview.net/pdf?id=HkuGJ3kCb p. 4
# def ppa(X, D):
#     X_mean = np.mean(X, axis=0)
#     X_sub = X - X_mean
#     us = PCA().fit_transform(X_sub)

#     X_out = np.zeros_like(X)
#     for i in tqdm(range(len(X))):
#         val = np.einsum('dj,j->d', us[:D], X[i])
#         val = np.einsum('d,dj->j', val, us[:D])
#         X_out[i] = X_sub[i] - val

#     return X_out


# # https://www.aclweb.org/anthology/W19-4328.pdf p. 237
# def dra(X, D):
#     X_1 = ppa(X, D)
#     X_2 = PCA(n_components=D).fit_transform(X_1)
#     X_3 = ppa(X_2, D)
#     return X_3


@click.command()
@click.option('--dim', default=300)
@click.option('--embed-files', default='conceptnet,glove')
@click.option('--outfile', default='embeddings-combined.txt')
def main(dim: int, embed_files: str, outfile: str):
    embed_files = [embed_file.lower() for embed_file in embed_files.split(',')]
    assert all(embed_file in EMBEDDING_FILES for embed_file in embed_files), \
        'Unknown embedding'

    print(f'Creating {dim} dim embeddings based on combining {embed_files}')
    embeds = [read_embeddings(f'embeddings-{embed_file}.txt')
              for embed_file in embed_files]
    vocab = get_vocab(*embeds)
    word_mat = concat_embeddings(vocab, *embeds)
    word_mat_reduced = reduce_dimensionality(word_mat, dim)
    word_mat_norm = normalize(word_mat_reduced)

    lines = [to_textfile_line(word, vec)
             for word, vec in zip(vocab, word_mat_norm)]

    print(f'Writing embeddings to {outfile}')
    with open(os.path.join('./', outfile), 'a') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
