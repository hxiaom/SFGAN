from base.base_data_loader import BaseDataLoader
from utils.tree import ClassNode

from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np
import os

class NsfcHierDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(NsfcHierDataLoader, self).__init__(config)
        self.data_df, self.class_tree = self.read_file()

        abstracts = self.data_df.iloc[:, 1].to_numpy()
        tags = self.data_df.iloc[:, 2].to_numpy()

        reviews = []
        labels = []
        texts = []

        for idx in range(abstracts.shape[0]):
            text = abstracts[idx]
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            reviews.append(sentences)
            labels.append(tags[idx])

        tokenizer = Tokenizer(num_words=self.config.data_loader.MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        # sequences = tokenizer.texts_to_sequences(texts)

        data = np.zeros((len(texts), self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')

        for i, sentences in enumerate(reviews):
            for j, sent in enumerate(sentences):
                if j < self.config.data_loader.MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if word in tokenizer.word_index:
                            if k < self.config.data_loader.MAX_SENT_LENGTH and tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS:
                                data[i, j, k] = tokenizer.word_index[word]
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
        self.X_test = None
        self.y_test = None

        GLOVE_DIR = "./data"
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
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

    def read_file(self):
        class_tree = ClassNode("ROOT",None,-1)
        hier_file = open(f"./data/label_hier.txt", 'r')
        contents = hier_file.readlines()
        cnt = 0
        for line in contents:
            line = line.split("\n")[0]
            line = line.split("\t")
            parent = line[0]
            children = line[1:]
            for child in children:
                parent_node = class_tree.find(parent)
                class_tree.find_add_child(parent, ClassNode(child, parent_node))
                cnt += 1
        
        # assign labels to classes in class tree
        offset = 0
        for i in range(1, class_tree.get_height()+1):
            nodes = class_tree.find_at_level(i)
            for node in nodes:
                node.label = offset
                offset += 1

        n_classes = class_tree.get_size() - 1
        print(f'Total number of classes: {n_classes}')
        print(class_tree.visualize_tree())
        
        data_df = pd.read_csv('./data/dataset.txt', sep='\t', header=None, names=['code', 'abstract'])
        data_df['code_num'] = data_df['code'].apply(class_tree.get_label)
        return data_df, class_tree
    

    def get_train_data(self):
        return self.X_train, self.y_train, len(self.word_index), self.embedding_matrix

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
        for i in range(len(self.data_df)):
            if self.data_df.iloc[i,]['code_num'] in child_code:
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

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_class_tree(self):
        return self.class_tree

    def get_embedding_matrix(self):
        return len(self.word_index), self.embedding_matrix
        