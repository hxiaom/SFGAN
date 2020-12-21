from base.base_data_loader import BaseDataLoader
from utils.tree import ClassNode

from nltk.tokenize import sent_tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np
import os

class NsfcDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(NsfcDataLoader, self).__init__(config)

    def get_train_data(self):
        data_df = pd.read_csv('./data/dataset_level2.txt', 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])
        code_to_index = {'A0101':0, 'A0102':1, 'A0103':2, 'A0104':3, 'A0105':4,
                        'A0106':5, 'A0107':6, 'A0108':7, 'A0109':8, 'A0110':9,
                        'A0111':10, 'A0112':11, 'A0113':12, 'A0114':13, 'A0115':14,
                        'A0116':15, 'A0117':16, 'A0201':17, 'A0202':18, 'A0203':19,
                        'A0204':20, 'A0205':21, 'A0206':22, 'A0301':23, 'A0302':24,
                        'A0303':25, 'A0304':26, 'A0305':27, 'A0306':28, 'A0307':29,
                        'A0308':30, 'A0309':31, 'A0310':32, 'A0401':33, 'A0402':34,
                        'A0403':35, 'A0404':36, 'A0405':37, 'A0501':38, 'A0502':39,
                        'A0503':40, 'A0504':41, 'A0505':42, 'A0506':43, 'A0507':44}
        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abstract_sents = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            code_index.append(code_to_index[data_df['code'][i]])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(abstracts)
        # sequences = tokenizer.texts_to_sequences(abstracts)


        data = np.zeros((abstract_num, self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        for i, abstract in enumerate(abstract_sents):
            for j, sent in enumerate(abstract):
                if j < self.config.data_loader.MAX_SENTS:
                    word_tokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(word_tokens):
                        if ((word in tokenizer.word_index) 
                                and (k < self.config.data_loader.MAX_SENT_LENGTH) 
                                and (tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS)):

                                data[i, j, k] = tokenizer.word_index[word]
                                k = k + 1

        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))

        # data = pad_sequences(sequences, maxlen=self.config.data_loader.MAX_SENTS)

        code_index = to_categorical(np.asarray(code_index))
        # labels = np.asarray(labels)
        print('Shape of X tensor:', data.shape)
        print('Shape of y tensor:', code_index.shape)

        self.X_train = data[:2984,:]
        self.y_train = code_index[:2984,:]
        self.X_test = data[2984:,:]
        self.y_test = code_index[2984:,:]

        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        embeddings_index = {}
        f = open('./data/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))

        self.embedding_matrix = np.random.random((len(self.word_index) + 1, self.config.data_loader.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        return self.X_train, self.y_train, self.X_test, self.y_test, len(self.word_index), self.embedding_matrix

    def get_train_data_plain(self):
        data_df = pd.read_csv('./data/dataset_level2.txt', 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])
        code_to_index = {'A0101':0, 'A0102':1, 'A0103':2, 'A0104':3, 'A0105':4,
                        'A0106':5, 'A0107':6, 'A0108':7, 'A0109':8, 'A0110':9,
                        'A0111':10, 'A0112':11, 'A0113':12, 'A0114':13, 'A0115':14,
                        'A0116':15, 'A0117':16, 'A0201':17, 'A0202':18, 'A0203':19,
                        'A0204':20, 'A0205':21, 'A0206':22, 'A0301':23, 'A0302':24,
                        'A0303':25, 'A0304':26, 'A0305':27, 'A0306':28, 'A0307':29,
                        'A0308':30, 'A0309':31, 'A0310':32, 'A0401':33, 'A0402':34,
                        'A0403':35, 'A0404':36, 'A0405':37, 'A0501':38, 'A0502':39,
                        'A0503':40, 'A0504':41, 'A0505':42, 'A0506':43, 'A0507':44}
        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abs = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            code_index.append(code_to_index[data_df['code'][i]])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(abstracts)
        # sequences = tokenizer.texts_to_sequences(abstracts)


        data = np.zeros((abstract_num, self.config.data_loader.MAX_DOC_LENGTH), dtype='int32')
        for i, abstract in enumerate(abstracts):
            word_tokens = text_to_word_sequence(abstract)
            j = 0
            for _, word in enumerate(word_tokens):
                if ((word in tokenizer.word_index) 
                        and (j < self.config.data_loader.MAX_DOC_LENGTH) 
                        and (tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS)):

                        data[i, j] = tokenizer.word_index[word]
                        j = j + 1

        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))

        # data = pad_sequences(sequences, maxlen=self.config.data_loader.MAX_SENTS)

        code_index = to_categorical(np.asarray(code_index))
        # labels = np.asarray(labels)
        print('Shape of X tensor:', data.shape)
        print('Shape of y tensor:', code_index.shape)

        self.X_train = data[:2984,:]
        self.y_train = code_index[:2984,:]
        self.X_test = data[2984:,:]
        self.y_test = code_index[2984:,:]

        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        embeddings_index = {}
        f = open('./data/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))

        self.embedding_matrix = np.random.random((len(self.word_index) + 1, self.config.data_loader.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        return self.X_train, self.y_train, self.X_test, self.y_test, len(self.word_index), self.embedding_matrix

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_data_by_code(self, code):
        child_code = []
        parent = self.class_tree.find(code)
        children = parent.children


        for child in children:
            if child.children == []:
                child_code.append(child.label)
            else:
                for c in child.children:
                    child_code.append(c.label)

        X_train_list = []
        y_list = []
        for i in range(len(self.X_train)):
            if self.data_df.iloc[i,]['tags'] in child_code:
                X_train_list.append(self.X_train[i])
                if child.children != []:
                    l = ord(self.data_df.iloc[i,]['code'][0]) - ord('A')
                else:
                    l = int(self.data_df.iloc[i,]['code'][1:3]) - 1
                y_list.append(l)
        X_train = np.asarray(X_train_list)
        y = to_categorical(np.asarray(y_list))
        data = [X_train, y]
        return data

    def get_test_data_by_code(self, code):
        child_code = []
        parent = self.class_tree.find(code)
        children = parent.children


        for child in children:
            if child.children == []:
                child_code.append(child.label)
            else:
                for c in child.children:
                    child_code.append(c.label)

        X_test_list = []
        y_list = []
        for i in range(33168, len(self.data_df)):
            if self.data_df.iloc[i,]['tags'] in child_code:
                X_test_list.append(self.X_test[i-33168])
                if child.children != []:
                    l = ord(self.data_df.iloc[i,]['code'][0]) - ord('A')
                else:
                    l = int(self.data_df.iloc[i,]['code'][1:3]) - 1
                y_list.append(l)
        X_test = np.asarray(X_test_list)
        y = to_categorical(np.asarray(y_list))
        data = [X_test, y]
        return data

    def get_train_data_by_level(self, level):
        if level == 0:
            return self.get_train_data_by_code('ROOT')
        else:
            child_code = []
            parent = self.class_tree.find('ROOT')
            children = parent.children

            y = np.zeros((len(self.X_train), 91), dtype='int32')
            X_train = []

            count = 0
            ind = 0
            for i, child in enumerate(children):
                X_train_temp, y_temp = self.get_train_data_by_code(child.name)
                X_train.append(X_train_temp)                
                y[count:count+y_temp.shape[0], ind:ind+y_temp.shape[1]] = y_temp
                count = count + y_temp.shape[0]
                ind = ind + y_temp.shape[1]

            X_train = np.concatenate(X_train, axis=0)
            return [X_train, y]

    def get_test_data_by_level(self, level):
        if level == 0:
            return self.get_test_data_by_code('ROOT')
        else:
            child_code = []
            parent = self.class_tree.find('ROOT')
            children = parent.children

            y = np.zeros((len(self.X_test), 91), dtype='int32')
            X_test = []

            count = 0
            ind = 0
            for i, child in enumerate(children):
                X_test_temp, y_temp = self.get_test_data_by_code(child.name)
                X_test.append(X_test_temp)                
                y[count:count+y_temp.shape[0], ind:ind+y_temp.shape[1]] = y_temp
                count = count + y_temp.shape[0]
                ind = ind + y_temp.shape[1]

            X_test = np.concatenate(X_test, axis=0)
            return [X_test, y]


    def get_embedding_matrix(self):
        return len(self.word_index), self.embedding_matrix
        