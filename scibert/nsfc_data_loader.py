from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
import os

def read_data():
    # self.file_name = './data/dataset_level2_test.txt'
    # self.split_index = 7807
    code_to_index = {'A0101':0, 'A0102':1, 'A0103':2, 'A0104':3, 'A0105':4,
                    'A0106':5, 'A0107':6, 'A0108':7, 'A0109':8, 'A0110':9,
                    'A0111':10, 'A0112':11, 'A0113':12, 'A0114':13, 'A0115':14,
                    'A0116':15, 'A0117':16, 'A0201':17, 'A0202':18, 'A0203':19,
                    'A0204':20, 'A0205':21, 'A0206':22, 'A0301':23, 'A0302':24,
                    'A0303':25, 'A0304':26, 'A0305':27, 'A0306':28, 'A0307':29,
                    'A0308':30, 'A0309':31, 'A0310':32, 'A0401':33, 'A0402':34,
                    'A0403':35, 'A0404':36, 'A0405':37, 'A0501':38, 'A0502':39,
                    'A0503':40, 'A0504':41, 'A0505':42, 'A0506':43, 'A0507':44}

    # self.file_name = './data/dataset_whole_level.txt'
    # self.split_index = 77607
    # self.code_to_index = {'A':0, 'B':1, 'C':2, 'D':3, 
    #                     'E':4, 'F':5, 'G':6, 'H':7}

    file_name = './data/A_multilabel.txt'
    split_index = 7074
    # split_index = 8

    # file_name = './data/A_multilabel_level2.txt'
    # split_index = 7074
    # code_to_index = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4}
    n_classes = 45
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
    sub_code_index = np.where(sub_code_index==1, 0.8, 0)
    code_index = np.add(main_code_index, sub_code_index)
    code_index[np.where(code_index >1)] = 1
    abstracts_train = abstracts[:split_index]
    abstracts_test = abstracts[split_index:]
    code_index_train = code_index[:split_index, :]
    code_index_test = code_index[split_index:, :]

    return abstracts_train, code_index_train, abstracts_test, code_index_test


def read_data_test():
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

    file_name = './data/A_multilabel_level2_test.txt'
    split_index = 7074
    code_to_index = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4}
    n_classes = 5
    MAX_SENTS = 30
    MAX_SENT_LENGTH = 100

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
    sub_code_index = np.where(sub_code_index==1, 0.8, 0)
    print('sub index', sub_code_index)
    code_index = np.add(main_code_index, sub_code_index)
    print('code index', code_index)
    code_index[np.where(code_index >1)] = 1
    print('code index', code_index)

    return abstracts, code_index