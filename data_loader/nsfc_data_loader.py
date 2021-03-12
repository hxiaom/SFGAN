from base.base_data_loader import BaseDataLoader

from nltk.tokenize import sent_tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle

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

        # self.file_name = './data/A_multilabel.txt'
        # self.split_index = 7074

        # self.file_name = './data/A_multilabel_level2.txt'
        # self.split_index = 7074
        # self.code_to_index = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4}
        # self.n_classes = 5

        self.file_name = './data/multilabel.txt'
        self.split_index = 7983
        self.code_to_index = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4,
                            'B01':5, 'B02':6, 'B03':7, 'B04':8, 'B05':9,
                            'B06':10, 'B07':11, 'B08': 12, 'C01':13, 'C02':14, 
                            'C03':15, 'C04':16, 'C05':17, 'C06':18, 'C07':19,
                            'C08':20, 'C09':21, 'C10':22, 'C11':23, 'C12':24,
                            'C13':25, 'C14':26, 'C15':27, 'C16':28, 'C17':29, 
                            'C18':30, 'C19':31, 'C20':32, 'C21':33, 'D01':34, 
                            'D02':35, 'D03':36, 'D04':37, 'D05':38, 'D06':39, 
                            'D07':40, 'E01':41, 'E02':42, 'E03':43, 'E04':44, 
                            'E05':45, 'E06':46, 'E07':47, 'E08':48, 'E09':49, 
                            'E10':50, 'E11':51, 'E12':52, 'E13':53, 'F01':54, 
                            'F02':55, 'F03':56, 'F04':57, 'F05':58, 'F06':59, 
                            'F07':60, 'G01':61, 'G02':62, 'G03':63, 'G04':64, 
                            'H01':65, 'H02':66, 'H03':67, 'H04':68, 'H05':69, 
                            'H06':70, 'H07':71, 'H08':72, 'H09':73, 'H10':74, 
                            'H11':75, 'H12':76, 'H13':77, 'H14':78, 'H15':79, 
                            'H16':80, 'H17':81, 'H18':82, 'H19':83, 'H20':84,
                            'H21':85, 'H22':86, 'H23':87, 'H24':88, 'H25':89, 
                            'H26':90, 'H27':91, 'H28':92, 'H29':93, 'H30':94, 
                            'H31':95}
        self.n_classes = 96


    def get_train_data(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'abstract', 'train_or_test'])

        abstract_sents = []
        code_index = []
        for i in range(len(data_df)):
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            code_index.append(self.code_to_index[data_df['code'][i]])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(data_df['abstract'].tolist())
        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))

        data = np.zeros((len(data_df), self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
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
        code_index = to_categorical(np.asarray(code_index))
        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]
        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        embeddings_index = {}
        f = open('./data/glove.6B.300d.txt')
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
        data_df = shuffle(data_df)

        abstract_sents = []   
        main_code_index = []
        sub_code_index = []
        for i in range(len(data_df)):
            abstract_sents.append(sent_tokenize(data_df['abstract'][i]))
            main_code_index.append(self.code_to_index[data_df['code'][i]])
            if data_df['sub_code'][i] == 'nan':
                sub_code_index.append(str(self.n_classes))
            else:
                sub_code_index.append(self.code_to_index[data_df['sub_code'][i]])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(data_df['abstract'].tolist())
        self.word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(self.word_index))
        data = np.zeros((len(data_df), self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
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
        
        main_code_index = to_categorical(np.asarray(main_code_index), num_classes=self.n_classes)
        sub_code_index = to_categorical(np.asarray(sub_code_index), num_classes=self.n_classes+1)
        sub_code_index = sub_code_index[:,:-1] # subcode has 'nan'
        sub_code_index = np.where(sub_code_index==1, 1, 0) # here can set subcode label as 0.8
        print('maincode label:\n', main_code_index)
        print('subcode label:\n', sub_code_index)
        code_index = np.add(main_code_index, sub_code_index)
        code_index[np.where(code_index >1)] = 1
        print('code label:\n', code_index)

        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]
        self.main_code_test_label = main_code_index[self.split_index:,:]
        self.sub_code_test_label = sub_code_index[self.split_index:,:]
        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        embeddings_index = {}
        f = open('./data/glove.6B.300d.txt')
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

        return (self.X_train, self.y_train, self.X_test, self.y_test, len(self.word_index), 
                self.embedding_matrix, self.main_code_test_label, self.sub_code_test_label)

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
        f = open('./data/glove.6B.300d.txt')
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

    def get_train_data_plain_multilabel(self):
        data_df = pd.read_csv(self.file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'sub_code', 'abstract', 'train_or_test'])
        data_df['sub_code'] = data_df['sub_code'].astype('str')
        data_df = shuffle(data_df, random_state=2)
        abstract_num = len(data_df)

        # codes = []
        # for i in range(abstract_num):
        #     # print(len(sent_tokenize(data_df['abstract'][i])))
        #     if data_df['sub_code'][i] == 'nan':
        #         codes.append((data_df['code'][i], ))
        #     else:
        #         codes.append((data_df['code'][i], data_df['sub_code'][i]))

        main_code_index = []
        sub_code_index = []
        for i in range(abstract_num):
            main_code_index.append(self.code_to_index[data_df['code'][i]])
            if data_df['sub_code'][i] == 'nan':
                sub_code_index.append(str(self.n_classes))
            else:
                sub_code_index.append(self.code_to_index[data_df['sub_code'][i]])
        
        abstracts = data_df['abstract'].tolist()

        abs = []
        code_index = []
        

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

        main_code_index = to_categorical(np.asarray(main_code_index), num_classes=self.n_classes)
        sub_code_index = to_categorical(np.asarray(sub_code_index), num_classes=self.n_classes+1)
        sub_code_index = sub_code_index[:,:-1]
        sub_code_index = np.where(sub_code_index==1, 1, 0)
        print('sub index', sub_code_index)
        code_index = np.add(main_code_index, sub_code_index)
        print('code index', code_index)
        code_index[np.where(code_index >1)] = 1
        print('code index', code_index)
        # one_hot = MultiLabelBinarizer()
        # code_index = one_hot.fit_transform(codes)
        # labels = np.asarray(labels)
        print('Shape of X tensor:', data.shape)
        print('Shape of y tensor:', code_index.shape)

        self.X_train = data[:self.split_index,:]
        self.y_train = code_index[:self.split_index,:]
        self.X_test = data[self.split_index:,:]
        self.y_test = code_index[self.split_index:,:]

        self.main_code_test_label = main_code_index[self.split_index:,:]
        self.sub_code_test_label = sub_code_index[self.split_index:,:]

        print('Shape of X_train tensor:', self.X_train.shape)
        print('Shape of y_train tensor:', self.y_train.shape)
        print('Shape of X_test tensor:', self.X_test.shape)
        print('Shape of y_test tensor:', self.y_test.shape)

        embeddings_index = {}
        f = open('./data/glove.6B.300d.txt')
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
        return self.X_train, self.y_train, self.X_test, self.y_test, len(self.word_index), self.embedding_matrix, self.main_code_test_label, self.sub_code_test_label

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


def get_data_bert():
    file_name = './data/multilabel.txt'
    split_index = 7983
    code_to_index = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4,
                        'B01':5, 'B02':6, 'B03':7, 'B04':8, 'B05':9,
                        'B06':10, 'B07':11, 'B08': 12, 'C01':13, 'C02':14, 
                        'C03':15, 'C04':16, 'C05':17, 'C06':18, 'C07':19,
                        'C08':20, 'C09':21, 'C10':22, 'C11':23, 'C12':24,
                        'C13':25, 'C14':26, 'C15':27, 'C16':28, 'C17':29, 
                        'C18':30, 'C19':31, 'C20':32, 'C21':33, 'D01':34, 
                        'D02':35, 'D03':36, 'D04':37, 'D05':38, 'D06':39, 
                        'D07':40, 'E01':41, 'E02':42, 'E03':43, 'E04':44, 
                        'E05':45, 'E06':46, 'E07':47, 'E08':48, 'E09':49, 
                        'E10':50, 'E11':51, 'E12':52, 'E13':53, 'F01':54, 
                        'F02':55, 'F03':56, 'F04':57, 'F05':58, 'F06':59, 
                        'F07':60, 'G01':61, 'G02':62, 'G03':63, 'G04':64, 
                        'H01':65, 'H02':66, 'H03':67, 'H04':68, 'H05':69, 
                        'H06':70, 'H07':71, 'H08':72, 'H09':73, 'H10':74, 
                        'H11':75, 'H12':76, 'H13':77, 'H14':78, 'H15':79, 
                        'H16':80, 'H17':81, 'H18':82, 'H19':83, 'H20':84,
                        'H21':85, 'H22':86, 'H23':87, 'H24':88, 'H25':89, 
                        'H26':90, 'H27':91, 'H28':92, 'H29':93, 'H30':94, 
                        'H31':95}
    n_classes = 96
    MAX_SENTS = 30
    MAX_SENT_LENGTH = 300

    data_df = pd.read_csv(file_name, 
                            sep='\t', 
                            header=None, 
                            names=['code', 'sub_code', 'abstract', 'train_or_test'])
    data_df['sub_code'] = data_df['sub_code'].astype('str')
    abstract_num = len(data_df)
    abstracts = data_df['abstract'].tolist()

    abstract_sents = []
    main_code_index = []
    sub_code_index = []
    for i in range(abstract_num):
        main_code_index.append(code_to_index[data_df['code'][i]])
        if data_df['sub_code'][i] == 'nan':
            sub_code_index.append(str(n_classes))
        else:
            sub_code_index.append(code_to_index[data_df['sub_code'][i]])


    main_code_index = to_categorical(np.asarray(main_code_index), num_classes=n_classes)
    sub_code_index = to_categorical(np.asarray(sub_code_index), num_classes=n_classes+1)
    sub_code_index = sub_code_index[:,:-1]
    sub_code_index = np.where(sub_code_index==1, 1, 0)
    code_index = np.add(main_code_index, sub_code_index)
    code_index[np.where(code_index >1)] = 1
    abstracts_train = abstracts[:split_index]
    abstracts_test = abstracts[split_index:]
    code_index_train = code_index[:split_index, :]
    code_index_test = code_index[split_index:, :]

    main_code_test_label = main_code_index[split_index:,:]
    sub_code_test_label = sub_code_index[split_index:,:]

    return abstracts_train, code_index_train, abstracts_test, code_index_test, main_code_test_label, sub_code_test_label
