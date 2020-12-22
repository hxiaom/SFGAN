from base.base_model import BaseModel

import tensorflow as tf
import tensorflow_addons as tfa
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


class FuncAttModel(BaseModel):
    def __init__(self, word_length, embedding_matrix, func_model, configs):
        super(FuncAttModel, self).__init__(configs)
        self.build_model(word_length, embedding_matrix, func_model)
        # self.model = []
        # # self.input_shape = (self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH)
        # self.x = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')

    def build_model(self, word_length, embedding_matrix, func_model):
        n_classes = 45
        embedding_layer = Embedding(word_length + 1,
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
        l_lstm_sent = Bidirectional(GRU(25, return_sequences=True, dropout=0.05))(review_encoder) # Query

        func_classification_model = Model(func_model.input, func_model.layers[-2].output)
        func_classification_model.trainable = False
        func_encoder = TimeDistributed(func_classification_model)(review_input) # Key

        query_value_attention_seq = Attention()([l_lstm_sent, review_encoder, func_encoder])
        query_encoding = GlobalAveragePooling1D()(
            func_encoder)
        query_value_attention = GlobalAveragePooling1D()(
            query_value_attention_seq)
        l_att_sent = Concatenate()(
            [query_encoding, query_value_attention])
        preds = Dense(n_classes, activation='softmax')(l_att_sent)
        self.model = Model(review_input, preds)
        
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', 
                        tf.keras.metrics.Recall(name='recall'), 
                        tf.keras.metrics.Precision(name='precision'),
                        tfa.metrics.F1Score(name='F1_micro', num_classes=45 ,average='micro'),
                        tfa.metrics.F1Score(name='F1_macro', num_classes=45 ,average='macro')])