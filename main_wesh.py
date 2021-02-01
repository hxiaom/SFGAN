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

from models.wesh_model import WeShModel

from trainers.wesh_trainer import WeShModelTrainer

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf

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
    X_train, y_train, X_test, y_test, word_length, embedding_matrix = data_loader.get_train_data()
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

if __name__ == '__main__':
    main()