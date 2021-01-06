from base.base_data_loader import BaseDataLoader
from utils.tree import ClassNode

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
import os

class NsfcDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(NsfcDataLoader, self).__init__(config)
        # self.file_name = './data/dataset_level2.txt'
        # self.split_index = 7807
        # self.code_to_index = {'A0101':0, 'A0102':1, 'A0103':2, 'A0104':3, 'A0105':4,
        #                 'A0106':5, 'A0107':6, 'A0108':7, 'A0109':8, 'A0110':9,
        #                 'A0111':10, 'A0112':11, 'A0113':12, 'A0114':13, 'A0115':14,
        #                 'A0116':15, 'A0117':16, 'A0201':17, 'A0202':18, 'A0203':19,
        #                 'A0204':20, 'A0205':21, 'A0206':22, 'A0301':23, 'A0302':24,
        #                 'A0303':25, 'A0304':26, 'A0305':27, 'A0306':28, 'A0307':29,
        #                 'A0308':30, 'A0309':31, 'A0310':32, 'A0401':33, 'A0402':34,
        #                 'A0403':35, 'A0404':36, 'A0405':37, 'A0501':38, 'A0502':39,
        #                 'A0503':40, 'A0504':41, 'A0505':42, 'A0506':43, 'A0507':44}
        
        # self.file_name = './data/dataset_whole_level.txt'
        # self.split_index = 77607
        # self.code_to_index = {'A':0, 'B':1, 'C':2, 'D':3, 
        #                     'E':4, 'F':5, 'G':6, 'H':7}

        self.file_name = './data/dataset_multilabel.txt'
        self.split_index = 779


    def get_train_data(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])
        
        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abstract_sents = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            code_index.append(self.code_to_index[data_df['code'][i]])

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

        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]

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

    def get_data_multilabel(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'sub_code', 'abstract', 'train_or_test'])
        data_df['sub_code'] = data_df['sub_code'].astype('str')
        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abstract_sents = []
        codes = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            if data_df['sub_code'][i] == 'nan':
                codes.append((data_df['code'][i], ))
            else:
                codes.append((data_df['code'][i], data_df['sub_code'][i]))

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

        one_hot = MultiLabelBinarizer()
        print(codes)
        code_index = one_hot.fit_transform(codes)

        # labels = np.asarray(labels)
        print('Shape of X tensor:', data.shape)
        print('Shape of y tensor:', code_index.shape)
        print('class name:', one_hot.classes_)

        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]

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

    def get_train_data_whole(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])

        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abstract_sents = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            code_index.append(self.code_to_index[data_df['code'][i]])

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

        self.X_train = data
        self.y_train = code_index

        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)

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
        return self.X_train, self.y_train, len(self.word_index), self.embedding_matrix

    def get_train_data_plain(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])

        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abs = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            code_index.append(self.code_to_index[data_df['code'][i]])

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

        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]

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

    def get_train_data_tfidf(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])

        abstract_num = len(data_df)

        abstracts = data_df['abstract'].tolist()

        abs = []
        code_index = []
        
        for i in range(abstract_num):
            # print(len(sent_tokenize(data_df['abstract'][i])))
            code_index.append(self.code_to_index[data_df['code'][i]])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(abstracts)
        # sequences = tokenizer.texts_to_sequences(abstracts)

        # TF-IDF Embedding
        tfidf = TfidfVectorizer(
            min_df = 100,
            max_df = 0.95,
            max_features = 8000,
            stop_words = 'english',
            # token_pattern=r"(?u)\S\S+"
        )
        print('train TF-IDF')
        tfidf.fit(data_df.abstract)
        print('transform TF-IDF')
        x_tfidf = tfidf.transform(data_df.abstract).toarray()
        y_tfidf = np.array(code_index)

        print('Shape of X tensor:', x_tfidf.shape)
        print('Shape of y tensor:', y_tfidf.shape)

        self.X_train = x_tfidf[:self.split_index,:]
        self.y_train = y_tfidf[:self.split_index,]
        self.X_test = x_tfidf[self.split_index:,:]
        self.y_test = y_tfidf[self.split_index:,]

        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        return self.X_train, self.y_train, self.X_test, self.y_test
