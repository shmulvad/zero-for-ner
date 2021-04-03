# CrossNER and Word Embeddings

## CrossNER pickle

The file [`CrossNER_NOUN_PRON.pickle`](./CrossNER_NOUN_PRON.pickle) is a pickled file containing a dictionary of dictionaries storing the nouns and pronouns from different domains. Below is some sample code showing how to read it and the structure:

```python
>>> import pickle
>>> with open('CrossNER_NOUN_PRON.pickle', 'rb') as f:
>>> 	ner_dict = pickle.load(f)
>>> 
>>> ner_dict.keys()
dict_keys(['ai', 'conll2003', 'literature', 'music', 'politics', 'science'])
>>> ner_dict['ai'].keys()
dict_keys(['NOUN', 'PRON'])
>>> ner_dict['ai']['NOUN'][:3]
[['model', 'NOUN', 'I-algorithm'],
 ['approaches', 'NOUN', 'O'],
 ['classifier', 'NOUN', 'I-algorithm']] 
```

Here is some sample code to loop over the obtained dictionary:

```python
for domain in ner_dict.keys():
    for pos in ner_dict[domain].keys():
        for word_info in ner_dict[topic][pos]:
            # Do something here
            # word_info is a list of length 3 similar to ['model', 'NOUN', 'I-algorithm']
```


## Word Embeddings

The word embeddings are stored in `embeddings-[SOURCE NAME].txt`. At the moment, ConceptNet and GloVe have been used as sources. They are stored in a similar format as i.e. GloVe (see below). This means each line contains the word at first and then 300 numbers representing the word vector. Based on the pickled file from CrossNER and the label words, there are 10545 embeddings at the moment.

```text
model 0.107 0.02 0.0305 ...
approaches 0.1324 0.0632 -0.0359 ...
⋮
```

Some sample code to read the file to a dictionary `word_embeddings`:

```python
import csv
import pandas as pd

def read_embeddings(embedding_file: str) -> Dict[str, np.ndarray]:
    words_df = pd.read_csv(embedding_file, sep=" ", index_col=0,
                           na_values=None, keep_default_na=False,
                           header=None, quoting=csv.QUOTE_NONE)
    word_embeddings = {word: words_df.loc[word].values
                       for word in words_df.index.values}
    return word_embeddings
    
    
word_embeddings = read_embeddings('embeddings-conceptnet.txt')
```

where `word_embeddings` is now a dictionary that can be queried by i.e. `word_embeddings['autoencoders']` to get the 300 dimensional word embedding.