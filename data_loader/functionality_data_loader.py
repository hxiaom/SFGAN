from base.base_data_loader import BaseDataLoader
from utils.tree import ClassNode

from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np
import os
import json

class FunctionalityDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(FunctionalityDataLoader, self).__init__(config)
        # df = pd.read_json('./data/functionality/train.jsonl')

        class_to_label = {'background':0, 'objective':1, 'method':2, 'result':3, 'other':4}
        texts = []
        reviews = []
        classes = []
        labels = []
        with open('./data/functionality/train_pubmed.jsonl') as f:
            for line in f:
                json_dict = json.loads(line)
                sentences_list = json_dict['sentences']
                texts += sentences_list
                for s in sentences_list:
                    reviews.append(tokenize.sent_tokenize(s))
                classes += json_dict['labels']

        for c in classes:
            labels.append(class_to_label[c])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        # sequences = tokenizer.texts_to_sequences(texts)

        data = np.zeros((len(texts), self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')


        for j, sent in enumerate(texts):
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word in tokenizer.word_index:
                    if k < self.config.data_loader.MAX_SENT_LENGTH and tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS:
                        data[j, k] = tokenizer.word_index[word]
                        k = k + 1

        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))

        # data = pad_sequences(sequences, maxlen=self.config.data_loader.MAX_SENTS)

        labels = to_categorical(np.asarray(labels))
        # labels = np.asarray(labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        self.X_train = data
        self.y_train = labels
        self.X_test = data
        self.y_test = labels

        embeddings_index = {}
        f = open('./data/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))

        # building Hierachical Attention network
        self.embedding_matrix = np.random.random((len(self.word_index) + 1, self.config.data_loader.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def get_train_data(self):
        return self.X_train, self.y_train, len(self.word_index), self.embedding_matrix
