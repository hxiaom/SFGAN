from data_loader.nsfc_data_loader import NsfcDataLoader
from models.textcnn_model import TextCNNModel
from trainers.textcnn_trainer import TextCNNModelTrainer
from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import recall_score, f1_score, hamming_loss, coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, average_precision_score, ndcg_score
import innvestigate
import innvestigate.utils as iutils

import datetime
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from matplotlib.font_manager import FontProperties

import time

MAX_SEQ_LENGTH = 400

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
    X_train, y_train = data_loader.get_data_plain_2()
    print("X_train\n", X_train)
    print("y_train\n", y_train)

    # create model
    num_classes = 91
    dropout_rate = 0.4
    EMBEDDING_DIM = 300

    docs_input = Input(shape=(400, 300))

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
    drop = Dropout(rate=dropout_rate)(flatten)
    # dense = Dense(300, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(drop)
    x_output = Dense(num_classes, kernel_initializer='he_uniform', activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(0.01))(drop)

    textcnn_model = Model(inputs=docs_input, outputs=x_output)
    textcnn_model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc', 
                    tf.keras.metrics.Recall(name='recall'), 
                    tf.keras.metrics.Precision(name='precision')])
    print(textcnn_model.summary())

    # train model
    history = textcnn_model.fit(
            X_train, y_train,
            epochs=2,
            # class_weight=class_weights,
            # verbose=self.config.trainer.verbose_training,
            batch_size=64,
            # validation_data = (self.data_test[0], self.data_test[1])
            validation_split=0.2,
        )

    textcnn_model.save_weights('./weight_6.h5')
    # textcnn_trainer.save()

    # Remove softmax layer
    model_with_softmax = textcnn_model
    # model_without_softmax = iutils.model_wo_softmax(textcnn_model.model)
    model_without_softmax = model_with_softmax
    print(model_without_softmax.summary())

    # Specify methods that you would like to use to explain the model. 
    # Please refer to iNNvestigate's documents for available methods.

    # methods = ['deep_taylor', 'gradient']
    methods = ['lrp.z']
    kwargs = [{}, {}, {}, {'pattern_type': 'relu'}]

    analyzer = innvestigate.create_analyzer('lrp.z', model_without_softmax)
    x = X_train[0]
    x = x.reshape((1, MAX_SEQ_LENGTH, 300))  
    presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
    print(presm) 
    a = analyzer.analyze(x)
    print('aaaaaaaaaaaaaaaaaaaaa')
    print(a)
    print(a['input_1'].shape)
    print('aaaaaaaaaaaaaaaaaaaaaaaaa')
    a = np.squeeze(a)
    a = np.sum(a, axis=1)
    print(a)
    aaaaaaa

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
        x = x.reshape((1, MAX_SEQ_LENGTH))   

        print("x type", type(x))
        print("x shape:", x.shape)
        presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
        prob = model_with_softmax.predict_on_batch(x)[0] #forward pass with softmax
        print("sample:", i, "\t output:", prob)
        y_hat = prob.argmax()
        test_sample_preds[i] = y_hat
        
        for aidx, analyzer in enumerate(analyzers):
            print(x)
            print(type(x))
            print(x.shape)
            # a = analyzer.analyze(x, neuron_selection=output_neuron)
            a = analyzer.analyze(x)
            print(a)
            a = np.squeeze(a)
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