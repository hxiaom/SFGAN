from data_loader.nsfc_data_loader import NsfcDataLoader

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger
from models.textcnn_model import TextCNNModel

from tensorflow.python.client import device_lib
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import recall_score, f1_score, hamming_loss, coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, average_precision_score, ndcg_score

import datetime
import sys
import numpy as np


def main():
    # capture the config and process the json configuration file
    try:
        args = get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.log_dir, config.callbacks.checkpoint_dir])

    # set logs
    sys.stdout = Logger(f'{config.callbacks.log_dir}/output.log', sys.stdout)
    sys.stderr = Logger(f'{config.callbacks.log_dir}/error.log', sys.stderr)

    # set GPU
    # if don't add this, it will report ERROR: Fail to find the dnn implementation.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return "No GPU available"
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
    print(device_lib.list_local_devices(),'\n')


    # load NSFC data
    print('Load NSFC data')
    data_loader = NsfcDataLoader(config)
    X_train, y_train, X_test, y_test, word_length, embedding_matrix = data_loader.get_data_plain()
    print("X_train\n", X_train)
    print("y_train\n", y_train)

    # create model
    textcnn_model = TextCNNModel(word_length, embedding_matrix, config)
    # textcnn_model = keras.models.load_model('./experiments/2021-03-24/default/checkpoints/default-35-1.82.hdf5')
    textcnn_model.model.load_weights('experiments/2021-04-07/textcnn/checkpoints/textcnn-85-2.23.hdf5')
    print(textcnn_model.model.summary())


    # Evaluation
    y_test_label = y_test.argmax(axis=-1)
    print('true result label')
    print(y_test_label)

    test_result = textcnn_model.model.predict(X_test)
    test_result_label = np.argmax(test_result, axis=1)
    print('test result label')
    print(test_result_label)

    cr = classification_report(y_test_label, test_result_label)
    print('cr', cr)

    # argsort method
    idxs = np.argsort(test_result, axis=1)[:,-2:]
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
 

if __name__ == '__main__':
    main()