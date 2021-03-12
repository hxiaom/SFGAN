from base.base_model import BaseModel

import tensorflow as tf
import tensorflow_addons as tfa
# from tensorflow_ranking.keras.losses import ListMLELoss
# from models.losses import plackett_luce_loss
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate, Masking
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, GlobalMaxPooling1D, BatchNormalization
from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.metrics import categorical_accuracy

from sklearn.metrics import label_ranking_average_precision_score

from time import time
import os
import numpy as np
import csv
import datetime

def listmle(x, t):
    """
    The ListMLE loss as in Xia et al (2008), Listwise Approach to Learning to Rank - Theory and Algorithm.
    """

    # Get the ground listtruth by sorting activations by the relevance labels
    t_hat = t[:, 0]
    x_hat = x[tf.reverse(tf.argsort(t_hat), axis=[0])]
    x_hat = K.cast(x_hat, tf.float32)

    # Compute MLE loss
    final = tf.reduce_logsumexp(x_hat)
    return K.sum(final - x_hat)

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

    # def get_config(self):  # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf/58680354#58680354
    #     config = super().get_config().copy()
    #     return config


class WeShModel(BaseModel):
    def __init__(self, word_length, embedding_matrix, configs):
        super(WeShModel, self).__init__(configs)
        self.n_classes = 96
        self.build_model(word_length, embedding_matrix)

    def build_model(self, word_length, embedding_matrix):
        
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
        # l_att = AttLayer(50)(l_lstm)
        l_att = GlobalMaxPooling1D()(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        proposal_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(proposal_input)
        l_lstm_sent = Bidirectional(GRU(50, return_sequences=True, dropout=0.3))(review_encoder)
        # l_att_sent = AttLayer(50)(l_lstm_sent)
        l_att_sent = GlobalMaxPooling1D()(l_lstm_sent)
        den = Dense(50, activation='relu')(l_att_sent)
        preds = Dense(self.n_classes, activation='sigmoid')(den)
        self.model = Model(proposal_input, preds)
        
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy', 
                        tf.keras.metrics.Recall(name='recall'), 
                        tf.keras.metrics.Precision(name='precision'),
                        tfa.metrics.F1Score(name='F1_micro', num_classes=self.n_classes, average='micro'),
                        tfa.metrics.F1Score(name='F1_macro', num_classes=self.n_classes, average='macro')])