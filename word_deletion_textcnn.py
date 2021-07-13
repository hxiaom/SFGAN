import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model

import innvestigate
import matplotlib.pyplot as plt

from utils import create_env_dir, get_data_word_deletion, plot_text_heatmap, set_gpu

EXP_NAME = 'textcnn'
MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 300
NUM_CLASSES = 91
DROPOUT_RATE = 0.4
NUM_EPOCHS = 1
FILE_NAME = './data/multilabel_copy.txt'
split_index = 7983
CODE_TO_INDEX = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4,
                'B01':5, 'B02':6, 'B03':7, 'B04':8, 'B05':9,
                'B06':10, 'B07':11, 'B08': 12, 'C01':13, 'C02':14, 
                'C03':15, 'C04':16, 'C05':17, 'C06':18, 'C07':19,
                'C08':20, 'C09':21, 'C10':22, 'C11':23, 'C12':24,
                'C13':25, 'C14':26, 'C15':27, 'C16':28, 'C17':29, 
                'C18':30, 'C19':31, 'C20':32, 'C21':33, 'D01':34, 
                'D02':35, 'D03':36, 'D04':37, 'D05':38, 'D06':39, 
                'D07':40, 'E01':41, 'E02':42, 'E03':43, 'E04':44, 
                'E05':45, 'E06':46, 'E07':47, 'E08':48, 'E09':49, 
                'F01':50, 'F02':51, 'F03':52, 'F04':53, 'F05':54, 
                'F06':55, 'G01':56, 'G02':57, 'G03':58, 'G04':59, 
                'H01':60, 'H02':61, 'H03':62, 'H04':63, 'H05':64, 
                'H06':65, 'H07':66, 'H08':67, 'H09':68, 'H10':69, 
                'H11':70, 'H12':71, 'H13':72, 'H14':73, 'H15':74, 
                'H16':75, 'H17':76, 'H18':77, 'H19':78, 'H20':79,
                'H21':80, 'H22':81, 'H23':82, 'H24':83, 'H25':84, 
                'H26':85, 'H27':86, 'H28':87, 'H29':88, 'H30':89, 
                'H31':90}

create_env_dir(EXP_NAME)

set_gpu()


# load NSFC data
print('Load NSFC data.')
X_train, main_code, sub_code, words = get_data_word_deletion(FILE_NAME, CODE_TO_INDEX)
y_train = main_code
print('NSFC data loaded.')
print("X_train shape\n", X_train.shape)
print("y_train shape\n", y_train.shape)


# build model
docs_input = Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM))
kernel_sizes = [3, 4, 5]
pooled = []
for kernel in kernel_sizes:
    conv = Conv1D(filters=100,
                kernel_size=kernel,
                padding='valid',
                strides=1,
                kernel_initializer='he_uniform',
                activation='relu')(docs_input)
    pool = MaxPooling1D(pool_size=400 - kernel + 1)(conv)
    pooled.append(pool)

merged = Concatenate(axis=-1)(pooled)
flatten = Flatten()(merged)
drop = Dropout(rate=DROPOUT_RATE)(flatten)
x_output = Dense(NUM_CLASSES, 
                kernel_initializer='he_uniform', 
                activation='sigmoid', 
                kernel_regularizer=tf.keras.regularizers.l1(0.01))(drop)

textcnn_model = Model(inputs=docs_input, outputs=x_output)
print(textcnn_model.summary())

# train model
textcnn_model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc', 
                tf.keras.metrics.Recall(name='recall'), 
                tf.keras.metrics.Precision(name='precision')])
# textcnn_model.load_weights('experiments/2021-05-11/textcnn_4/checkpoints/textcnn_4-40-3.34.hdf5', by_name=True, skip_mismatch=True)
textcnn_model.load_weights('weight_6.h5', by_name=True, skip_mismatch=True)


# LRP methods
# Remove softmax layer
model_with_softmax = textcnn_model
# model_without_softmax = iutils.model_wo_softmax(textcnn_model.model)
model_without_softmax = model_with_softmax
print(model_without_softmax.summary())

x_correct = []
y_correct = []
lrp_correct = []
x_wrong = []
y_wrong = []

analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_without_softmax)

# for i in range(len(X_train)):
for i in range(500):
    x = X_train[i]
    x = x.reshape((1, MAX_SEQ_LENGTH, 300))  

    presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
    idxs = np.argsort(presm, axis=0)
    neuron_list = [idxs[-1], idxs[-2]]

    idxs_true = np.argmax(y_train[i])

    if idxs_true in neuron_list:
        x_correct.append(x)
        y_correct.append(y_train[i])
        lrp_score = analyzer.analyze(x, neuron_selection=int(idxs_true))
        lrp_score = lrp_score['input_1']
        lrp_score = np.squeeze(lrp_score)
        lrp_score = np.sum(lrp_score, axis=1)
        relevance_rank = np.argsort(lrp_score, axis=0)
        lrp_correct.append(relevance_rank)

print(x_correct)
print(len(y_correct))
print(lrp_correct[-1])

acc = []
for j in range(50): 
    count = 0
    for i in range(len(y_correct)):
        lrp = lrp_correct[i]
        x_correct[i][0, lrp[-j]] = 0
        print('delete ',lrp[-j], 'th word')
        # print(x_correct[i][0, lrp[-j]])
        print('zero num', i, np.sum(x_correct[i]==0))
        presm = model_without_softmax.predict_on_batch(x_correct[i])[0] #forward pass without softmax
        idxs = np.argsort(presm, axis=0)
        neuron_list = [idxs[-1], idxs[-2]]
        idxs_true = np.argmax(y_correct[i])
        print('True label', idxs_true)
        print('Predict label', neuron_list)
        if idxs_true in neuron_list:
            count = count + 1
            print('Correct num:', count)
    print('--------------------------')
    print('delete %s words' % j)
    print('Total delete word num: ', count)
    print('acc:', count/len(y_correct))
    acc.append(count/len(y_correct))

print('acc', acc)
# # ['lrp', 'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 
# # 'lrp.flat', 'lrp.alpha_beta', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 
# # 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 'lrp.z_plus_fast', 
# # 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 
# # 'lrp.sequential_preset_b_flat', 'lrp.rule_until_index']
# analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_without_softmax)
# # analyzer = innvestigate.create_analyzer('sa', model_without_softmax)

# x = X_train[0]
# x = x.reshape((1, MAX_SEQ_LENGTH, 300))  

# presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
# print(presm)
# # argsort method
# idxs = np.argsort(presm, axis=0)
# print(idxs)

# idxs_true = np.argmax(y_train[0])
# print(idxs_true)

# neuron_list = [idxs[-1], idxs[-2]]

# if idxs_true in neuron_list:
#     print('True')

# # for i, neuron in enumerate(neuron_list):
# #     print(i)
# #     print(neuron)
# #     neuron = int(neuron)
# #     lrp_score = analyzer.analyze(x, neuron_selection=neuron)
# #     lrp_score = lrp_score['input_1']
# #     lrp_score = np.squeeze(lrp_score)
# #     lrp_score = np.sum(lrp_score, axis=1)
# #     print(lrp_score)
# #     print(lrp_score.shape)

# #     # words = ['test']*400
# #     plot_text_heatmap(words[0], lrp_score.reshape(-1), title='Method: %s' % 'lrp', verbose=0)
# #     plt.savefig('./plot%s_%s.png' % (i, neuron), format='png')
# #     plt.show()

# lrp_score = analyzer.analyze(x, neuron_selection=int(idxs_true))
# lrp_score = lrp_score['input_1']
# lrp_score = np.squeeze(lrp_score)
# lrp_score = np.sum(lrp_score, axis=1)
# print(lrp_score)
# print(lrp_score.shape)

# relevance_rank = np.argsort(lrp_score, axis=0)
# print(relevance_rank)

# x[0, relevance_rank[-1]] = 0

# presm_deletion = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
# print(presm[0])
# print(presm_deletion[0])