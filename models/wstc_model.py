from keras.engine.topology import Layer, InputSpec
from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Attention, GlobalAveragePooling1D
from keras import initializers
from keras import backend as K
from keras.models import Model
from time import time

import os
import numpy as np
import csv

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

class FuncAttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(FuncAttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        # self.trainable_weights = [self.W, self.b, self.u]
        self.trainable_weights.append([self.W, self.b])
        super(FuncAttLayer, self).build(input_shape)

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

# class WSTC(object):
class NsfcHierModel(BaseModel):
    def __init__(self, config, class_tree):
        super(NsfcHierModel, self).__init__(config)
        self.class_tree = class_tree
        self.model = []
        self.eval_set = None
        self.sup_dict = {}
        self.block_label = {}
        self.siblings_map = {}
        self.block_level = 1
        self.block_thre = 1.0
        self.input_shape = (self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH)
        self.x = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')

    def FganModel(self, n_classes, word_index_length, embedding_matrix):
        embedding_layer = Embedding(word_index_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=True,
                                    mask_zero=True)

        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        l_att = AttLayer(50)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)

        new_model = Model(self.func_model.input, self.func_model.layers[-2].output)
        print(new_model.summary())
        func_encoder = TimeDistributed(new_model)(review_input)

        a = GRU(50, return_sequences=True)(review_encoder)
        l_lstm_sent = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
        
        # l_att_sent = AttLayer(50)(l_lstm_sent)
        query_value_attention_seq = Attention()([l_lstm_sent, func_encoder])
        query_encoding = GlobalAveragePooling1D()(
            func_encoder)
        query_value_attention = GlobalAveragePooling1D()(
            query_value_attention_seq)
        l_att_sent = Concatenate()(
            [query_encoding, query_value_attention])
        preds = Dense(n_classes, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)
        return model

    def FunctionalityModel(self, word_index_length, embedding_matrix):
        embedding_layer = Embedding(word_index_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=True,
                                    mask_zero=True)
        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        flat = Flatten()(lstm)
        dense = Dense(100, activation='relu')(flat)
        preds = Dense(5, activation='softmax')(dense)
        model = Model(sentence_input, preds)
        return model

    def train_func(self, X, y, length, matrix):
        self.func_model = self.FunctionalityModel(length, matrix)
        self.func_model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
        self.func_model.fit(X, y)
        for k,v in self.func_model._get_trainable_state().items():
            k.trainable = False


    def instantiate(self, class_tree, word_index_length, embedding_matrix):
        num_children = len(class_tree.children)
        print('number of children', num_children)
        if num_children <= 1:
            class_tree.model = None
        else:
            class_tree.model = self.FganModel(num_children, word_index_length, embedding_matrix)

    def pretrain(self, data, model):
        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
        t0 = time()
        print('\nPretraining...')
        model.fit(data[0], data[1], batch_size=64, epochs=1)
        print(f'Pretraining time: {time() - t0:.2f}s')
        save_dir='experiments'
        suffix='1'
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save_weights(f'{save_dir}/pretrained_{suffix}.h5')

    def ensemble_classifier(self, level, class_tree):
        if level == 0:
            result_classifier = class_tree.model
        elif level == 1:
            children = class_tree.children
            outputs = []
            for i, child in enumerate(children):
                a = IndexLayer(i)(class_tree.model(self.x))
                b = child.model(self.x)
                print(a.shape)
                print(b.shape)
                c = Multiply()([a, b])
                outputs.append(c)
            z = Concatenate()(outputs)
            result_classifier = Model(inputs=self.x, outputs=z)
            print(result_classifier)
        return result_classifier


    def extract_label(self, y, level):
        if type(level) is int:
            relevant_nodes = self.class_tree.find_at_level(level)
            relevant_labels = [relevant_node.label for relevant_node in relevant_nodes]
        else:
            relevant_labels = []
            for i in level:
                relevant_nodes = self.class_tree.find_at_level(i)
                relevant_labels += [relevant_node.label for relevant_node in relevant_nodes]
        if type(y) is dict:
            y_ret = {}
            for key in y:
                y_ret[key] = y[key][relevant_labels]
        else:
            y_ret = y[:, relevant_labels]
        return y_ret


    def expand_pred(self, q_pred, level, cur_idx):
        y_expanded = np.zeros((self.input_shape[0], q_pred.shape[1]))
        print(y_expanded.shape)
        if level not in self.siblings_map:
            self.siblings_map[level] = self.class_tree.siblings_at_level(level)
        siblings_map = self.siblings_map[level]
        block_idx = []
        for i, q in enumerate(q_pred):
            pred = np.argmax(q)
            idx = cur_idx[i]
            if level >= self.block_level and self.block_thre < 1.0 and idx not in self.sup_dict:
                siblings = siblings_map[pred]
                siblings_pred = q[siblings]/np.sum(q[siblings])
                if len(siblings) >= 2:
                    conf_val = entropy(siblings_pred)/np.log(len(siblings))
                else:
                    conf_val = 0
                if conf_val > self.block_thre:
                    block_idx.append(idx)
                else:
                    y_expanded[idx,pred] = 1.0
            else:
                y_expanded[idx,pred] = 1.0
        if self.block_label:
            blocked = [idx for idx in self.block_label]
            blocked_labels = np.array([label for label in self.block_label.values()])
            blocked_labels = self.extract_label(blocked_labels, level+1)
            y_expanded[blocked,:] = blocked_labels
        return y_expanded, block_idx


    def compile(self, level, optimizer='sgd', loss='kld'):
        self.model[level].compile(optimizer=optimizer, loss=loss)


    def fit(self, data, level, maxiter=5e4, batch_size=256, tol=0.1, power=2,
            update_interval=100, save_dir='experiments', save_suffix=''):
        model = self.model[level]
        print(f'Update interval: {update_interval}')

        model.fit(data[0], data[1], batch_size=64, epochs=1)
        return []

def IndexLayer(idx):
    def func(x):
        return x[:, idx]

    return Lambda(func)