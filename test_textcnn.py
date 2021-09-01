from tensorflow.python.client import device_lib
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import recall_score, f1_score, hamming_loss, coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, average_precision_score, ndcg_score
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model


import datetime
import sys
import numpy as np
from utils import create_env_dir, plot_text_heatmap, get_data_plain, set_gpu, get_data_multilabel

MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 300
NUM_CLASSES = 91
DROPOUT_RATE = 0.4
NUM_EPOCHS = 10
EXP_NAME = 'textcnn'
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
print('Load NSFC data')
X_test, y_test, word_length, embedding_matrix, words = get_data_multilabel(FILE_NAME, CODE_TO_INDEX)
print("X_train\n", X_test)
print("y_train\n", y_test)

# create model
docs_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedding_layer = Embedding(word_length + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False,
                            name='embedding_test'
                            # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                            )
embedded_docs = embedding_layer(docs_input)
kernel_sizes = [3, 4, 5]
pooled = []
for kernel in kernel_sizes:
    conv = Conv1D(filters=100,
                kernel_size=kernel,
                padding='valid',
                strides=1,
                kernel_initializer='he_uniform',
                activation='relu')(embedded_docs)
    pool = MaxPooling1D(pool_size=400 - kernel + 1)(conv)
    pooled.append(pool)
merged = Concatenate(axis=-1)(pooled)
flatten = Flatten()(merged)
x_output = Dense(NUM_CLASSES, 
                name='dense_new',
                activation='sigmoid', 
                kernel_regularizer=tf.keras.regularizers.l1(0.01))(flatten)

textcnn_model = Model(inputs=docs_input, outputs=x_output)
print(textcnn_model.summary())


# textcnn_model = keras.models.load_model('./experiments/2021-03-24/default/checkpoints/default-35-1.82.hdf5')
textcnn_model.load_weights('finetune_weights.h5', by_name=True, skip_mismatch=True)


# Evaluation
y_test_label = y_test.argmax(axis=-1)
print('true result label')
print(y_test_label)

test_result = textcnn_model.predict(X_test)
test_result_label = np.argmax(test_result, axis=1)
print('test result label')
print(test_result_label)


# argsort method
idxs = np.argsort(test_result, axis=1)[:,-2:]
test_result_label = test_result
test_result_label.fill(0)
for i in range(idxs.shape[0]):
    for j in range(idxs.shape[1]):
        # if test_result[i][idxs[i][j]] >= 0.5:
        test_result_label[i][idxs[i][j]] = 1
# test_true = y_test.argmax(axis=-1)

cr = classification_report(y_test, test_result_label)
print('cr', cr)


count = 0
for i in range(len(y_test_label)):
    if test_result_label[i][y_test_label[i]] == 1:
        count = count + 1

percent = count / len(y_test_label)
print('top two accuracy:')
print(percent)

# argsort method
idxs = np.argsort(test_result, axis=1)[:,-3:]
test_result_label = test_result
test_result_label.fill(0)
for i in range(idxs.shape[0]):
    for j in range(idxs.shape[1]):
        # if test_result[i][idxs[i][j]] >= 0.5:
        test_result_label[i][idxs[i][j]] = 1
# test_true = y_test.argmax(axis=-1)

count = 0
for i in range(len(y_test_label)):
    if test_result_label[i][y_test_label[i]] == 1:
        count = count + 1

percent = count / len(y_test_label)
print('top three accuracy:')
print(percent)

dataset = open('./wrong_answer.txt', 'a')
dataset.write('index' + 'true' + '\t' + 'predict' + '\n')
for i in range(len(y_test_label)):
    if y_test_label[i] != test_result_label[i]:
        dataset.write(str(i) + '\t' + str(y_test_label[i]) + '\t' + str(test_result_label[i]) + '\n')
dataset.close()