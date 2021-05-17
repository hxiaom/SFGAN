from data_loader.nsfc_data_loader import NsfcDataLoader
from models.textcnn_model import TextCNNModel
from trainers.textcnn_trainer import TextCNNModelTrainer
from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf
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
        print("No GPU available")
        return
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
    print(textcnn_model.model.summary())

    # train model
    textcnn_trainer = TextCNNModelTrainer(textcnn_model.model, [X_train, y_train], [X_test, y_test], config)
    textcnn_trainer.train()

    print(type(textcnn_model.model))
    # textcnn_model.model.save('./model.h5')
    textcnn_model.model.save_weights('./weight_4.h5')
    # textcnn_trainer.save()

    # Evaluation
    y_train_label = y_train.argmax(axis=-1)
    print('true result label')
    print(y_train_label)

    train_result = textcnn_model.model.predict(X_train)
    train_result_label = np.argmax(train_result, axis=1)
    print('train result label')
    print(train_result_label)

    cr = classification_report(y_train_label, train_result_label)
    print('cr', cr)

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

    precision = precision_score(y_test_label, test_result_label, average=None)
    precision_micro = precision_score(y_test_label, test_result_label, average='micro')
    print('Precision:', precision_micro)
    print(precision)

    # Recall
    recall = recall_score(y_test_label, test_result_label, average=None)
    recall_micro = recall_score(y_test_label, test_result_label, average='micro')
    print('Recall:', recall_micro)
    print(recall)

    # F1_score
    F1 = f1_score(y_test_label, test_result_label, average=None)
    F1_micro = f1_score(y_test_label, test_result_label, average='micro')
    print('F1:', F1_micro)
    print(F1)

if __name__ == '__main__':
    main()