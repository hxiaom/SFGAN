from base.base_data_loader import BaseDataLoader
import pandas as pd
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import os


class NsfcDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(NsfcDataLoader, self).__init__(config)
        data_train = pd.read_csv('./data/dataset.txt', sep='\t', header=None)
        abstracts = data_train.iloc[:, 1].to_numpy()
        tags = data_train.iloc[:, 0].to_numpy()

        tags_to_number = {}
        tags_unique = np.unique(tags)
        for i in range(len(tags_unique)):
            tags_to_number[tags_unique[i]] = i

        reviews = []
        labels = []
        texts = []

        for idx in range(abstracts.shape[0]):
            text = abstracts[idx]
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            reviews.append(sentences)
            labels.append(tags_to_number[tags[idx]])

        tokenizer = Tokenizer(nb_words=self.config.data_loader.MAX_NB_WORDS)
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

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(self.config.trainer.validation_split * data.shape[0])

        self.X_train = data[:-nb_validation_samples]
        self.y_train = labels[:-nb_validation_samples]
        self.X_test = data[-nb_validation_samples:]
        self.y_test = labels[-nb_validation_samples:]


    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_embedding_matrix(self):
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

        return len(self.word_index), self.embedding_matrix