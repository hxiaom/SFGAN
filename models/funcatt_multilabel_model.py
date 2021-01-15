from base.base_model import BaseModel

import tensorflow as tf
import tensorflow_addons as tfa
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate, Masking
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Attention, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers import AdditiveAttention
from keras import initializers
from keras import backend as K
from keras.models import Model
from models.attention import AttentionLayer

from time import time
import os
import numpy as np
import csv
import datetime


class FuncAttModel(BaseModel):
    def __init__(self, word_length, embedding_matrix, func_model, configs):
        super(FuncAttModel, self).__init__(configs)
        self.n_classes = 45
        self.build_model(word_length, embedding_matrix, func_model)

    def build_model(self, word_length, embedding_matrix, func_model):
        
        embedding_layer = Embedding(word_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )
        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(50, return_sequences=True, dropout=0.3))(embedded_sequences)
        l_att = GlobalMaxPooling1D()(l_lstm)
        # l_att = AttLayer(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)  # Value
        l_lstm_sent = Bidirectional(GRU(50, return_sequences=True, dropout=0.3))(review_encoder)
        l_att_sent = GlobalMaxPooling1D()(l_lstm_sent)
        # l_att_sent = AttLayer(100)(l_lstm_sent) # Key

        # func_classification_model = Model(func_model.input, func_model.layers[-2].output)
        # func_classification_model.trainable = False
        # func_encoder = TimeDistributed(func_classification_model)(review_input) # Query

        x = func_model.layers[-4](embedded_sequences)
        x = func_model.layers[-3](x)
        x = func_model.layers[-2](x)
        x = func_model.layers[-1](x)
        func_classification_model = Model(sentence_input, x)
        func_classification_model.trainable = False
        func_encoder = TimeDistributed(func_classification_model, name='func')(review_input) # Query

        att_layer = AttentionLayer(name='attention_layer')
        query_value_attention_seq, att_states = att_layer([review_encoder, func_encoder])
        print('l_att_sent - output shape:', l_att_sent.shape)
        print('l_lstm_sent - output shape:', l_lstm_sent.shape)
        print('review_encoder - output shape:', review_encoder.shape)
        print('func_encoder - output shape:', func_encoder.shape)
        print('query_value_attention_seq  - shape', query_value_attention_seq.shape)
        print('att_states - shape', att_states.shape)
        
        # query_encoding = GlobalAveragePooling1D()(
        #     func_encoder)
        # func_encoder = Flatten()(func_encoder)
        query_value_attention = GlobalAveragePooling1D()(
            query_value_attention_seq)
        con = Concatenate()(
            [l_att_sent, query_value_attention])
        con = Dense(50, activation='relu')(con)
        preds = Dense(self.n_classes, activation='sigmoid')(con)
        self.model = Model(review_input, preds)
        
        self.model.compile(loss='binary_crossentropy',
            #   loss_weights = [1.0, 0.0],
              optimizer='adam',
              metrics=['acc', 
                        tf.keras.metrics.Recall(name='recall'), 
                        tf.keras.metrics.Precision(name='precision'),
                        tfa.metrics.F1Score(name='F1_micro', num_classes=self.n_classes, average='micro'),
                        tfa.metrics.F1Score(name='F1_macro', num_classes=self.n_classes, average='macro')])

def custom_loss_function(y_true, y_pred):
    return 0