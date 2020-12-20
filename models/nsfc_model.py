from base.base_model import BaseModel

import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate, Masking
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Attention, GlobalAveragePooling1D, BatchNormalization
from keras import initializers
from keras import backend as K
from keras.models import Model

from time import time
import os
import numpy as np
import csv
import datetime


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        # self.trainable_weights = [self.W, self.b, self.u]
        self.trainable_weights.append([self.W, self.b, self.u])
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class NsfcModel(BaseModel):
    def __init__(self, configs):
        super(NsfcModel, self).__init__(configs)
        self.model = []
        # self.input_shape = (self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH)
        self.x = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')

    def SfganModel(self, n_classes, word_index_length, embedding_matrix, func_model):
        embedding_layer = Embedding(word_index_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )

        # embedding_layer = Masking(mask_value=0)(embedding_layer)
        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        batch_nor = BatchNormalization()(embedded_sequences)
        l_lstm = Bidirectional(GRU(25, return_sequences=True))(batch_nor)
        l_att = AttLayer(25)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)
        # c = GlobalAveragePooling1D()(embedded_sequences)
        # d = Dense(50)(c)
        # sentEncoder = Model(sentence_input, d)

        review_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)  # Value
        l_lstm_sent = Bidirectional(GRU(25, return_sequences=True))(review_encoder) # Query

        func_classification_model = Model(func_model.input, func_model.layers[-2].output)
        func_encoder = TimeDistributed(func_classification_model)(review_input) # Key

        query_value_attention_seq = Attention()([l_lstm_sent, review_encoder, func_encoder])
        query_encoding = GlobalAveragePooling1D()(
            func_encoder)
        query_value_attention = GlobalAveragePooling1D()(
            query_value_attention_seq)
        l_att_sent = Concatenate()(
            [query_encoding, query_value_attention])
        preds = Dense(n_classes, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)
        return model

    def SfganModel_without_functionality(self, n_classes, word_index_length, embedding_matrix):
        embedding_layer = Embedding(word_index_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )

        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(25, return_sequences=True, dropout=0.05))(embedded_sequences)
        l_att = AttLayer(25)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(25, return_sequences=True, dropout=0.05))(review_encoder)
        l_att_sent = AttLayer(25)(l_lstm_sent)
        preds = Dense(n_classes, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)

        return model

    def FunctionalityModel(self, word_index_length, embedding_matrix):
        embedding_layer = Embedding(word_index_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )
        # embedding_layer = Masking(mask_value=0)(embedding_layer)
        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        flat = Flatten()(lstm)
        dense = Dense(50, activation='relu')(flat)
        preds = Dense(5, activation='softmax')(dense)
        model = Model(sentence_input, preds)
        return model

    def train_func_classification_model(self, X, y, length, matrix):
        self.func_model = self.FunctionalityModel(length, matrix)
        self.func_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        self.func_model.fit(X, y,
                            batch_size=self.config.func_trainer.batch_size,
                            # batch_size=1, 
                            epochs=self.config.func_trainer.num_epochs,
                            validation_split=self.config.func_trainer.validation_split
                            )
        for k,v in self.func_model._get_trainable_state().items():
            k.trainable = False
        self.func_model.save_weights(f'{self.config.callbacks.checkpoint_dir}/functionality.h5')
        return self.func_model
        
    def load_func_model(self, length, matrix):
        self.func_model = self.FunctionalityModel(length, matrix)
        self.func_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        self.func_model.load_weights('./experiments/functionality.h5')
        return self.func_model

    def instantiate(self, class_tree, word_index_length, embedding_matrix):
        num_children = len(class_tree.children)
        print('number of children', num_children)
        if num_children <= 1:
            class_tree.model = None
        else:
            class_tree.model = self.SfganModel(num_children, word_index_length, embedding_matrix)
            # class_tree.model = self.SfganModel_without_functionality(num_children, word_index_length, embedding_matrix)

    def pretrain(self, data, data_test, model):
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        t0 = time()
        print('\nPretraining...')
        model.fit(data[0], 
                data[1], 
                batch_size=self.config.local_trainer.batch_size, 
                epochs=self.config.local_trainer.num_epochs,
                # validation_split=self.config.local_trainer.validation_split,
                validation_data = (data_test[0], data_test[1]))
        print(f'Pretraining time: {time() - t0:.2f}s')

        model.save_weights(f'{self.config.callbacks.checkpoint_dir}/pretrained_func_classification.h5')

    def ensemble_classifier(self, level, class_tree):
        if level == 0:
            result_classifier = class_tree.model
        elif level == 1:
            children = class_tree.children
            outputs = []
            top_level_model = class_tree.model(self.x)
            for i, child in enumerate(children):
                print('ensemble node', child.name)
                a = IndexLayer(i)(top_level_model)
                b = child.model(self.x)
                c = Multiply()([a, b])
                outputs.append(c)
            z = Concatenate()(outputs)
            result_classifier = Model(inputs=self.x, outputs=z)
            print(result_classifier)
        return result_classifier

    def compile(self, level, optimizer='adam', loss='categorical_crossentropy'):
        self.model[level].compile(optimizer=optimizer, loss=loss, metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    def fit(self, data, data_test, level):
        model = self.model[level]
        print('start fitting', datetime.datetime.now())
        model.fit(data[0], 
                data[1], 
                batch_size=self.config.global_trainer.batch_size, 
                epochs=self.config.global_trainer.num_epochs,
                # validation_split=self.config.global_trainer.validation_split,
                validation_data = (data_test[0], data_test[1]))

        print('finish fitting', datetime.datetime.now())
        model.save_weights(f'{self.config.callbacks.checkpoint_dir}/{level}.h5')
        print('finish save model', datetime.datetime.now())
        return

def IndexLayer(idx):
    def func(x):
        return x[:, idx]
    return Lambda(func)
