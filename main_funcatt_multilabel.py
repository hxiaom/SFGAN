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
experiment.add_tag('funcatt')

from data_loader.nsfc_data_loader import NsfcDataLoader
from data_loader.functionality_data_loader import FunctionalityDataLoader

from models.func_model import FuncModel
from models.funcatt_multilabel_model import FuncAttModel

from trainers.funcatt_trainer import FuncAttModelTrainer

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.models import Model

from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss, average_precision_score, ndcg_score

import datetime
import sys
import numpy as np

class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color



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

    # load functionality model
    word_length_func = 136411
    embedding_matrix_func = np.loadtxt('./experiments/embedding_matrix_func_200.txt')
    func_model = FuncModel(word_length_func, embedding_matrix_func, config)
    func_model.load_model()
    print(func_model.model.summary())

    # create model
    funcatt_model = FuncAttModel(word_length, embedding_matrix, func_model.model, config)
    print(funcatt_model.model.summary())

    # train model
    funcatt_trainer = FuncAttModelTrainer(funcatt_model.model, [X_train, y_train], [X_test, y_test], config)
    funcatt_trainer.train()

    # # create functionality output model
    # layer_name = 'func'
    # func_output_model = Model(inputs=funcatt_trainer.model.input,
    #                         outputs=funcatt_trainer.model.get_layer(layer_name).output)
    # last_layer = func_model.model.layers[-1](func_output_model.output)
    # func_output = Model(inputs=func_output_model.input, outputs=last_layer)
    # test_output = func_output.predict(X_test)
    # print(np.argmax(test_output[-1], axis=1))

    # func_file = open('./experiments/functionality_output.txt', 'w')
    # func_output = np.argmax(test_output, axis=2).tolist()
    # func_file.writelines(["%s\n" % item  for item in func_output])

    # model = funcatt_model.model
    # model.summary()
    # model = Model(inputs=model.input,
    #             outputs=[model.output, model.get_layer('attention_layer').output])

    # # if you are using batches the outputs will be in batches
    # # get exact attentions of chars
    # an_attention_output = attention_outputs[0][-len(encoded_input_text):]

    # # before the prediction i supposed you tokenized text
    # # you need to match each char and attention
    # char_vals = [CharVal(c, v) for c, v in zip(tokenized_text, attention_output)]
    # import pandas as pd
    # char_df = pd.DataFrame(char_vals).transpose()
    # # apply coloring values
    # char_df = char_df.style.applymap(color_charvals)
    # char_df

    # Evaluation
    test_result = funcatt_model.model.predict(X_test)
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
    # Partitions Evaluation
    # Precision
    precision = precision_score(y_test_label, test_result_label, average=None)
    precision_macro = precision_score(y_test_label, test_result_label, average='micro')
    print('Precision:', precision_macro)
    print(precision)

    # Recall
    recall = recall_score(y_test_label, test_result_label, average=None)
    recall_macro = recall_score(y_test_label, test_result_label, average='micro')
    print('Recall:', recall_macro)
    print(recall)

    # F1_score
    F1 = f1_score(y_test_label, test_result_label, average=None)
    F1_macro = f1_score(y_test_label, test_result_label, average='micro')
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