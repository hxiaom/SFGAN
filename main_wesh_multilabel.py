# version: 2020.12.20
from comet_ml import Experiment
experiment = Experiment(
    project_name="proposalclassification",
    workspace="hxiaom",
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)
experiment.add_tag('wesh')

from data_loader.nsfc_data_loader import NsfcDataLoader

from models.wesh_multilabel_model import WeShModel

from trainers.wesh_trainer import WeShModelTrainer

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf

import datetime
import sys
import numpy as np

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, hamming_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss, average_precision_score, ndcg_score


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
    X_train, y_train, X_test, y_test, word_length, embedding_matrix = data_loader.get_data_multilabel()
    print("X_train\n", X_train)
    print("y_train\n", y_train)

    # X_train, y_train, word_length, embedding_matrix = data_loader.get_train_data_whole()
    # print("X_train\n", X_train)
    # print("y_train\n", y_train)

    # create model
    wesh_model = WeShModel(word_length, embedding_matrix, config)
    print(wesh_model.model.summary())

    # train model
    # wesh_trainer = WeShModelTrainer(wesh_model.model, [X_train, y_train], None, config)
    wesh_trainer = WeShModelTrainer(wesh_model.model, [X_train, y_train], [X_test, y_test], config)

    wesh_trainer.train()

    # Evaluation
    test_result = wesh_model.model.predict(X_test)
    test_result_label = test_result
    # threshold method
    test_result_label[test_result_label>=0.5] = 1
    test_result_label[test_result_label<0.5] = 0

    print(test_result)
    print(y_test)

    # # argsort method
    # idxs = np.argsort(test_result, axis=1)[:,-2:]
    # test_result_label = test_result
    # test_result_label.fill(0)
    # for i in range(idxs.shape[0]):
    #     for j in range(idxs.shape[1]):
    #         # if test_result[i][idxs[i][j]] >= 0.5:
    #         test_result_label[i][idxs[i][j]] = 1
    # # test_true = y_test.argmax(axis=-1)

    # print(idxs)
    # print(test_result)
    # print(y_test)

    y_test_label = y_test
    y_test_label[y_test_label>=0.5] = 1

    cr = classification_report(y_test_label, test_result_label)
    print('cr', cr)
    # Partitions Evaluation
    # Precision
    precision = precision_score(y_test_label, test_result_label, average=None)
    precision_macro = precision_score(y_test_label, test_result_label, average='micro')
    print('Precision:', precision)
    print('Precision:', precision_macro)
    print(precision)

    # Recall
    recall = recall_score(y_test_label, test_result_label, average=None)
    recall_macro = recall_score(y_test_label, test_result_label, average='micro')
    print('Recall:', recall)
    print('Recall:', recall_macro)
    print(recall)

    # F1_score
    F1 = f1_score(y_test_label, test_result_label, average=None)
    F1_macro = f1_score(y_test_label, test_result_label, average='micro')
    print('F1:', F1)
    print('F1:', F1_macro)
    print(F1)

    # Hamming Loss
    hamming = hamming_loss(y_test_label, test_result_label)
    print('Hamming Loss:', hamming)

    # Rankings Evaluation
    # Coverage
    coverage = coverage_error(y_test, test_result)
    print('Coverage Error:', coverage)

    # Average Precision Score
    lrap = average_precision_score(y_test, test_result)
    print('Average Precision Score:', lrap)

    # Ranking Loss
    rl = label_ranking_loss(y_test, test_result)
    print('Ranking Loss:', rl)

    # ndcg_score
    ndcg = ndcg_score(y_test, test_result)
    print('NDCG', ndcg)

if __name__ == '__main__':
    main()