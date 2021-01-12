from data_loader.nsfc_data_loader import NsfcDataLoader

from models.wesh_multilabel_model import WeShModel, AttLayer

from trainers.wesh_trainer import WeShModelTrainer

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf
import keras

from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss

import datetime
import sys
import numpy as np


def main():
    # capture the config and process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
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
    X_train, y_train, X_test, y_test, word_length, embedding_matrix = data_loader.get_data_multilabel()
    print("X_train\n", X_train)
    print("y_train\n", y_train)

    # X_train, y_train, word_length, embedding_matrix = data_loader.get_train_data_whole()
    # print("X_train\n", X_train)
    # print("y_train\n", y_train)

    # create model
    wesh_model = keras.models.load_model('./experiments/2021-01-12/wesh_1/checkpoints/wesh_1-53-0.05.hdf5')
    print(wesh_model.summary())


    # train model
    # wesh_trainer = WeShModelTrainer(wesh_model.model, [X_train, y_train], None, config)
    test_result = wesh_model.predict(X_test)
    
    # threshold method
    test_result[test_result>=0.5] = 1
    test_result[test_result<0.5] = 0

    y_test[y_test>=0.5] = 1
    y_test[y_test<0.5] = 0

    print(test_result)
    print(y_test)

    # # argsort method
    # test_result = test_result.argmax(axis=-1)
    # idxs = np.argsort(proba)[::-1][:2]
    # test_true = y_test.argmax(axis=-1)
    # print(test_result)
    # print(test_true)

    # Partitions Evaluation
    # Precision
    precision = precision_score(y_test, test_result, average=None)
    precision_macro = precision_score(y_test, test_result, average='macro')
    print('Precision:', precision_macro)
    print(precision)

    # Recall
    recall = recall_score(y_test, test_result, average=None)
    recall_macro = recall_score(y_test, test_result, average='macro')
    print('Recall:', recall_macro)
    print(recall)

    # F1_score
    F1 = f1_score(y_test, test_result, average=None)
    F1_macro = f1_score(y_test, test_result, average='macro')
    print('F1:', F1_macro)
    print(F1)

    # Hamming Loss
    hamming = hamming_loss(y_test, test_result)
    print('Hamming Loss:', hamming)

    # Rankings Evaluation
    # Coverage
    coverage = coverage_error(y_test, test_result)
    print('Coverage Error:', coverage)

    # Average Precision Score
    lrap = label_ranking_average_precision_score(y_test, test_result)
    print('Average Precision Score:', lrap)

    # Ranking Loss
    rl = label_ranking_loss(y_test, test_result)
    print('Ranking Loss:', rl)

if __name__ == '__main__':
    main()