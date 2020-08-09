from keras.engine.topology import Layer, InputSpec
from data_loader.nsfc_data_loader import NsfcDataLoader

from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
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
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_att = AttLayer(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(self.config.data_loader.MAX_SENTS, self.config.data_loader.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        print(review_encoder)
        a = GRU(100, return_sequences=True)(review_encoder)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
        l_att_sent = AttLayer(100)(l_lstm_sent)
        preds = Dense(n_classes, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)
        return model
    
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

    def ensemble(self, class_tree, level, parent_output):
        outputs = []
        if class_tree.model:
            print("in the ensemble")
            y_curr = class_tree.model(self.x)
            if parent_output is not None:
                y_curr = Multiply()([parent_output, y_curr])
        else:
            y_curr = parent_output

        if level == 0:
            outputs.append(y_curr)
        else:
            for i, child in enumerate(class_tree.children):
                outputs += self.ensemble(child, level - 1, IndexLayer(i)(y_curr))
        return outputs

    def ensemble_classifier(self, level, class_tree):
        outputs = self.ensemble(class_tree, level,  None)
        print(outputs)
        print(len(outputs[0].get_shape()))
        outputs = [ExpanLayer(-1)(output) if len(output.get_shape()) < 2 else output for output in outputs]
        print('new')
        print(outputs)
        z = Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
        return Model(inputs=self.x, outputs=z)

    # def __init__(self,
    #              input_shape,
    #              class_tree,
    #              max_level,
    #              sup_source,
    #              init=RandomUniform(minval=-0.01, maxval=0.01),
    #              y=None,
    #              vocab_sz=None,
    #              word_embedding_dim=100,
    #              blocking_perc=0,
    #              block_thre=1.0,
    #              block_level=1,
    #              ):

    #     super(WSTC, self).__init__()

    #     self.input_shape = input_shape
    #     self.class_tree = class_tree
    #     self.y = y
    #     if type(y) == dict:
    #         self.eval_set = np.array([ele for ele in y])
    #     else:
    #         self.eval_set = None
    #     self.vocab_sz = vocab_sz
    #     self.block_level = block_level
    #     self.block_thre = block_thre
    #     self.block_label = {}
    #     self.siblings_map = {}
    #     self.x = Input(shape=(input_shape[1],), name='input')
    #     self.model = []
    #     self.sup_dict = {}
    #     if sup_source == 'docs':
    #         n_classes = class_tree.get_size() - 1
    #         leaves = class_tree.find_leaves()
    #         for leaf in leaves:
    #             current = np.zeros(n_classes)
    #             for i in class_tree.name2label(leaf.name):
    #                 current[i] = 1.0
    #             for idx in leaf.sup_idx:
    #                 self.sup_dict[idx] = current

    # def ensemble(self, class_tree, level, input_shape, parent_output):
    #     outputs = []
    #     if class_tree.model:
    #         y_curr = class_tree.model(self.x)
    #         if parent_output is not None:
    #             y_curr = Multiply()([parent_output, y_curr])
    #     else:
    #         y_curr = parent_output

    #     if level == 0:
    #         outputs.append(y_curr)
    #     else:
    #         for i, child in enumerate(class_tree.children):
    #             outputs += self.ensemble(child, level - 1, input_shape, IndexLayer(i)(y_curr))
    #     return outputs

    # def ensemble_classifier(self, level):
    #     outputs = self.ensemble(self.class_tree, level, self.input_shape[1], None)
    #     outputs = [ExpanLayer(-1)(output) if len(output.get_shape()) < 2 else output for output in outputs]
    #     z = Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
    #     return Model(inputs=self.x, outputs=z)

    

    # def load_weights(self, weights, level):
    #     print(f'Loading weights @ level {level}')
    #     self.model[level].load_weights(weights)

    # def load_pretrain(self, weights, model):
    #     model.load_weights(weights)

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

    # def predict(self, x, level):
    #     q = self.model[level].predict(x, verbose=0)
    #     return q.argmax(1)

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

    # def aggregate_pred(self, q_all, level, block_idx, cur_idx, agg="All"):
    #     leaves = self.class_tree.find_at_level(level+1)
    #     leaves_labels = [leaf.label for leaf in leaves]
    #     parents = self.class_tree.find_at_level(level)
    #     parents_labels = [parent.label for parent in parents]
    #     ancestor_dict = {}
    #     for leaf in leaves:
    #         ancestors = leaf.find_ancestors()
    #         ancestor_dict[leaf.label] = [ancestor.label for ancestor in ancestors]
    #     for parent in parents:
    #         ancestors = parent.find_ancestors()
    #         ancestor_dict[parent.label] = [ancestor.label for ancestor in ancestors]
    #     y_leaf = np.argmax(q_all[:, leaves_labels], axis=1)
    #     y_leaf = [leaves_labels[y] for y in y_leaf]
    #     if level > 0:
    #         y_parents = np.argmax(q_all[:, parents_labels], axis=1)
    #         y_parents = [parents_labels[y] for y in y_parents]
    #     if agg == "Subset" and self.eval_set is not None:
    #         cur_eval = [ele for ele in self.eval_set if ele in cur_idx]
    #         inv_cur_idx = {i:idx for idx, i in enumerate(cur_idx)}
    #         y_aggregate = np.zeros((len(cur_eval), q_all.shape[1]))
    #         for i, raw_idx in enumerate(cur_eval):
    #             idx = inv_cur_idx[raw_idx]
    #             if raw_idx not in block_idx:
    #                 y_aggregate[i, y_leaf[idx]] = 1.0
    #                 for ancestor in ancestor_dict[y_leaf[idx]]:
    #                     y_aggregate[i, ancestor] = 1.0
    #             else:
    #                 if level > 0:
    #                     y_aggregate[i, y_parents[idx]] = 1.0
    #                     for ancestor in ancestor_dict[y_parents[idx]]:
    #                         y_aggregate[i, ancestor] = 1.0
    #     else:
    #         y_aggregate = np.zeros((self.input_shape[0], q_all.shape[1]))
    #         for i in range(len(q_all)):
    #             idx = cur_idx[i]
    #             if idx not in block_idx:
    #                 y_aggregate[idx, y_leaf[i]] = 1.0
    #                 for ancestor in ancestor_dict[y_leaf[i]]:
    #                     y_aggregate[idx, ancestor] = 1.0
    #             else:
    #                 if level > 0:
    #                     y_aggregate[idx, y_parents[i]] = 1.0
    #                     for ancestor in ancestor_dict[y_parents[i]]:
    #                         y_aggregate[idx, ancestor] = 1.0
    #         if self.block_label:
    #             blocked = [idx for idx in self.block_label]
    #             blocked_labels = np.array([label for label in self.block_label.values()])
    #             blocked_labels = self.extract_label(blocked_labels, range(1, level+2))
    #             y_aggregate[blocked, :] = blocked_labels
    #     return y_aggregate

    # def record_block(self, block_idx, y_pred_agg):
    #     n_classes = self.class_tree.get_size() - 1
    #     for idx in block_idx:
    #         self.block_label[idx] = np.zeros(n_classes)
    #         self.block_label[idx][:len(y_pred_agg[idx])] = y_pred_agg[idx]

    # def target_distribution(self, q, nonblock, sup_level, power=2):
    #     q = q[nonblock]
    #     weight = q ** power / q.sum(axis=0)
    #     p = (weight.T / weight.sum(axis=1)).T
    #     inv_nonblock = {k:v for v,k in enumerate(nonblock)}
    #     for i in sup_level:
    #         mapped_i = inv_nonblock[i]
    #         p[mapped_i] = sup_level[i]
    #     return p

    def compile(self, level, optimizer='sgd', loss='kld'):
        self.model[level].compile(optimizer=optimizer, loss=loss)
        # print(f"\nLevel {level} model summary: ")
        # self.model[level].summary()

    def fit(self, data, level, maxiter=5e4, batch_size=256, tol=0.1, power=2,
            update_interval=100, save_dir='experiments', save_suffix=''):
        model = self.model[level]
        print(f'Update interval: {update_interval}')
        
        cur_idx = np.array([idx for idx in range(data[0].shape[0]) if idx not in self.block_label])
        # x = x[cur_idx]
        # y = self.y
        x = data[0]
        y = data[1]

        # logging files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfiles = []
        logwriters = []
        for i in range(level+2):
            if i <= level:
                logfile = open(save_dir + f'/self_training_log_level_{i}{save_suffix}.csv', 'w')
            else:
                logfile = open(save_dir + f'/self_training_log_all{save_suffix}.csv', 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'f1_macro', 'f1_micro'])
            logwriter.writeheader()
            logfiles.append(logfile)
            logwriters.append(logwriter)

        index = 0

        if y is not None:
            if self.eval_set is not None:
                cur_eval = [idx for idx in self.eval_set if idx in cur_idx]
                y = np.array([y[idx] for idx in cur_eval])
            y_all = []
            label_all = []
            for i in range(level+1):
                y_curr = self.extract_label(y, i+1)
                y_all.append(y_curr)
                nodes = self.class_tree.find_at_level(i+1)
                label_all += [node.label for node in nodes]
            y = y[:, label_all]

        mapped_sup_dict_level = {}
        if len(self.sup_dict) > 0:
            sup_dict_level = self.extract_label(self.sup_dict, level+1)
            inv_cur_idx = {i:idx for idx, i in enumerate(cur_idx)}        
            for key in sup_dict_level:
                mapped_sup_dict_level[inv_cur_idx[key]] = sup_dict_level[key]
    
        # for ite in range(int(maxiter)):
        #     try:
        #         if ite % update_interval == 0:
        #             print(f'\nIter {ite}: ')
        #             y_pred_all = []
        #             q_all = np.zeros((len(x), 0))
        #             for i in range(level+1):
        #                 q_i = self.model[i].predict(x)
        #                 q_all = np.concatenate((q_all, q_i), axis=1)
        #                 y_pred_i, block_idx = self.expand_pred(q_i, i, cur_idx)
        #                 y_pred_all.append(y_pred_i)
        #             q = q_i
        #             y_pred = y_pred_i
        #             if len(block_idx) > 0:
        #                 print(f'Number of blocked documents back to level {level}: {len(block_idx)}')
        #             y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)

        #             if y is not None:
        #                 if self.eval_set is not None:
        #                     y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx, agg="Subset")
        #                     y_pred_all = [y_pred[cur_eval, :] for y_pred in y_pred_all]
        #                     for i in range(level+1):
        #                         f1_macro, f1_micro = np.round(f1(y_all[i], y_pred_all[i]), 5)
        #                         print(f'Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}')
        #                         logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
        #                         logwriters[i].writerow(logdict)
        #                     f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
        #                     logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
        #                     logwriters[-1].writerow(logdict)
        #                     print(f'Evaluated at subset of size {len(cur_eval)}: f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes')
        #                 else:
        #                     y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)
        #                     for i in range(level+1):
        #                         f1_macro, f1_micro = np.round(f1(y_all[i], y_pred_all[i]), 5)
        #                         print(f'f1_macro = {f1_macro}, f1_micro = {f1_micro} @ level {i+1}')
        #                         logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
        #                         logwriters[i].writerow(logdict)
        #                     f1_macro, f1_micro = np.round(f1(y, y_pred_agg), 5)
        #                     logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
        #                     logwriters[-1].writerow(logdict)
        #                     print(f'f1_macro = {f1_macro}, f1_micro = {f1_micro} @ all classes')
                        
        #             nonblock = np.array(list(set(range(x.shape[0])) - set(block_idx)))
        #             x_nonblock = x[nonblock]
        #             p_nonblock = self.target_distribution(q, nonblock, mapped_sup_dict_level, power)

        #             if ite > 0:
        #                 change_idx = []
        #                 for i in range(len(y_pred)):
        #                     if not np.array_equal(y_pred[i], y_pred_last[i]):
        #                         change_idx.append(i)
        #                 y_pred_last = np.copy(y_pred)
        #                 delta_label = len(change_idx)
        #                 print(f'Fraction of documents with label changes: {np.round(delta_label/y_pred.shape[0]*100, 3)} %')
                        
        #                 if delta_label/y_pred.shape[0] < tol/100:
        #                     print(f'\nFraction: {np.round(delta_label / y_pred.shape[0] * 100, 3)} % < tol: {tol} %')
        #                     print('Reached tolerance threshold. Self-training terminated.')
        #                     break
        #             else:
        #                 y_pred_last = np.copy(y_pred)

        #         # train on batch
        #         index_array = np.arange(x_nonblock.shape[0])
        #         if index * batch_size >= x_nonblock.shape[0]:
        #             index = 0
        #         idx = index_array[index * batch_size: min((index + 1) * batch_size, x_nonblock.shape[0])]
        #         try:
        #             assert len(idx) > 0
        #         except AssertionError:
        #             print(f'Error @ index {index}')
        #         model.train_on_batch(x=x_nonblock[idx], y=p_nonblock[idx])
        #         index = index + 1 if (index + 1) * batch_size < x_nonblock.shape[0] else 0
        #         ite += 1

        #     except KeyboardInterrupt:
        #         print("\nKeyboard interrupt! Self-training terminated.")
        #         break

        # for logfile in logfiles:
        #     logfile.close()

        # if save_dir is not None:
        #     model.save_weights(save_dir + '/final.h5')
        #     print(f"Final model saved to: {save_dir}/final.h5")
        # q_all = np.zeros((len(x), 0))
        # for i in range(level+1):
        #     q_i = self.model[i].predict(x)
        #     q_all = np.concatenate((q_all, q_i), axis=1)
        # y_pred_agg = self.aggregate_pred(q_all, level, block_idx, cur_idx)
        # self.record_block(block_idx, y_pred_agg)
        # return y_pred_agg
        return []

def IndexLayer(idx):
    def func(x):
        return x[:, idx]

    return Lambda(func)


def ExpanLayer(dim):
    def func(x):
        return K.expand_dims(x, dim)

    return Lambda(func)