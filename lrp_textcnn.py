from data_loader.nsfc_data_loader import NsfcDataLoader

from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger
from models.textcnn_model import TextCNNModel

from tensorflow.python.client import device_lib
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import recall_score, f1_score, hamming_loss, coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, average_precision_score, ndcg_score

import datetime
import time
import sys
import numpy as np

import innvestigate
import innvestigate.utils as iutils
from innvestigate.utils.tests.networks import base as network_base

# plot package
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from matplotlib.font_manager import FontProperties

MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 300

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
    print(textcnn_model.model.summary())
    # textcnn_model = keras.models.load_model('./experiments/2021-03-24/default/checkpoints/default-35-1.82.hdf5')
    # textcnn_model.model.load_weights('experiments/2021-04-07/textcnn/checkpoints/textcnn-85-2.23.hdf5')
    textcnn_model.model.load_weights('experiments/2021-05-11/textcnn_4/checkpoints/textcnn_4-40-3.34.hdf5')
    

    # Evaluation
    y_test_label = y_test.argmax(axis=-1)
    print('true result label')
    print(y_test_label)

    test_result = textcnn_model.model.predict(X_test)
    labels_pred = np.argmax(test_result, axis=1)
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

    # dataset = open('./wrong_answer.txt', 'a')
    # dataset.write('index' + 'true' + '\t' + 'predict' + '\n')
    # for i in range(len(y_test_label)):
    #     if y_test_label[i] != test_result_label[i]:
    #         dataset.write(str(i) + '\t' + str(y_test_label[i]) + '\t' + str(test_result_label[i]) + '\n')
    # dataset.close()
 

    # Remove softmax layer
    model_with_softmax = textcnn_model.model
    # model_without_softmax = iutils.model_wo_softmax(textcnn_model.model)
    model_without_softmax = model_with_softmax
    model_without_softmax.summary()

    # Specify methods that you would like to use to explain the model. 
    # Please refer to iNNvestigate's documents for available methods.

    # methods = ['deep_taylor', 'gradient']
    methods = ['deep_taylor']
    kwargs = [{}, {}, {}, {'pattern_type': 'relu'}]

    # build an analyzer for each method
    analyzers = []

    for method, kws in zip(methods, kwargs):
        print(method)
        analyzer = innvestigate.create_analyzer(method, model_without_softmax, neuron_selection_mode="index", **kws)
        analyzers.append(analyzer)

    # specify cluster that we want to investigate
    inspect_cluster = 4
    output_neuron = inspect_cluster

    test_sample_indices = [i for i, j in enumerate(labels_pred) if j == inspect_cluster]
    test_sample_preds = [None]*len(test_sample_indices)

    # a variable to store analysis results.
    analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, MAX_SEQ_LENGTH])

    # interpret each sample using each method
    for i, ridx in enumerate(test_sample_indices):
        # get sample
        t_start = time.time()
        x = X_train[ridx]
        x = x.reshape((1, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM))   


        presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
        prob = model_with_softmax.predict_on_batch(x)[0] #forward pass with softmax
        print("sample:", i, "\t output:", prob)
        y_hat = prob.argmax()
        test_sample_preds[i] = y_hat
        
        for aidx, analyzer in enumerate(analyzers):
            a = np.squeeze(analyzer.analyze(x, neuron_selection=output_neuron))
            a = np.sum(a, axis=1)
            analysis[i, aidx] = a
            print(a)
        t_elapsed = time.time() - t_start
        print('Review %d (%.4fs)'% (ridx, t_elapsed))

    # Traverse over the analysis results and visualize them.
    for i, idx in enumerate(test_sample_indices):

        # Only show first 10 samples
        if i == 5:
            break
        
        words = paper_df.iloc[idx,:]['content_token'][:MAX_SEQ_LENGTH]
                                    
        for j, method in enumerate(methods):
            print(len(words))
            print(len(analysis[i, j].reshape(-1)))
            plot_text_heatmap(words, analysis[i, j].reshape(-1), title='Method: %s' % method, verbose=0)
            plt.show()

# This is a utility method visualizing the relevance scores of each word to the network's prediction. 
# one might skip understanding the function, and see its output first.
def getChineseFont():  
    return FontProperties(fname='./SimHei.ttf',size=16) 

def plot_text_heatmap(words, scores, title="", width=10, height=0.3, verbose=0, max_word_per_line=15, font_size=30):
    '''plot text heatmap

    Args:
        words:
        scores:
        title:
        width:
        height:
        verbose:
        max_word_per_line:
    
    Returns:
        None
    '''
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(0.2+normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        if normalized_scores[i] < 0.6:
            color = 'black'
        
        text = ax.text(0.0, loc_y, token, 
                    fontsize=font_size,
                    color=color,
                    transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            word_sapce = font_size * 0.7
            t = transforms.offset_copy(text._transform, x=ex.width+word_sapce, units='dots')

    if verbose == 0:
        ax.axis('off')
    
    return

if __name__ == '__main__':
    main()