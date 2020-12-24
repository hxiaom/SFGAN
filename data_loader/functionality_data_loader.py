from base.base_data_loader import BaseDataLoader

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np
import json

class FunctionalityDataLoader(BaseDataLoader):
    def __init__(self, config):
        # load data
        super(FunctionalityDataLoader, self).__init__(config)


    def get_train_data(self):
        functionality_to_index = {'background':0, 'objective':1, 'method':2, 'result':3, 'other':4}
        abstract_sents = []
        functionality = []
        func_index = []

        # load data from jsonl
        # train_pubmed.jsonl
        # {"abstract_id": "24562799", "sentences": ["Many pathogenic ...", "It was ..."], 
        #                             "labels": ["background", "background"], 
        #                             "confs": [1, 1]}
        with open('./data/functionality/train_pubmed.jsonl') as f:
            for line in f:
                json_dict = json.loads(line)
                abstract_sents += json_dict['sentences']
                functionality += json_dict['labels']

        # transfer functionality to index
        for f in functionality:
            func_index.append(functionality_to_index[f])

        # 0-1 encoding of func index
        func_index = to_categorical(np.asarray(func_index))

        # traning tokenizor
        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(abstract_sents)
        # sequences = tokenizer.texts_to_sequences(abstract_sents)

        # token embedding matrix, [sentence_number, MAX_SENT_LENGTH]
        data = np.zeros((len(abstract_sents), self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        for i, sent in enumerate(abstract_sents):
            word_tokens = text_to_word_sequence(sent)
            j = 0   # the j th words in a sentence
            for _, word in enumerate(word_tokens):
                if word in tokenizer.word_index and j < self.config.data_loader.MAX_SENT_LENGTH and tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS:
                        data[i, j] = tokenizer.word_index[word]
                        j = j + 1

        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))

        # data = pad_sequences(sequences, maxlen=self.config.data_loader.MAX_SENTS)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', func_index.shape)

        self.X_train = data
        self.y_train = func_index
        self.X_test = data
        self.y_test = func_index

        # load glove matrix
        embeddings_index = {}
        f = open('./data/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Load glove data.')
        # glove embedding matrix
        self.embedding_matrix = np.random.random((len(self.word_index) + 1, self.config.data_loader.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        np.savetxt('./experiments/embedding_matrix_func_100.txt', self.embedding_matrix)

        return self.X_train, self.y_train, len(self.word_index), self.embedding_matrix
