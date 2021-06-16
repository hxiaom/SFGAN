import time
import sys
import os
import re

import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def create_env_dir(EXP_NAME):
    # create the experiments dirs
    log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), EXP_NAME, "logs/")
    checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), EXP_NAME, "checkpoints/")
    create_dirs([log_dir, checkpoint_dir])

    # set logs
    sys.stdout = Logger(f'{log_dir}/output.log', sys.stdout)
    sys.stderr = Logger(f'{log_dir}/error.log', sys.stderr)


def plot_text_heatmap(words, scores, title="", width=15, height=7, verbose=0, max_word_per_line=20):
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    # ax.set_title(title, loc='left')
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
    loc_y = 1.1

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(-0.1, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 0,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  0.05
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')


def tokenize_and_remove_stop_words(text):
    '''tokenize and remove stop words

    Args: 
        string

    Returns:
        token list as ['token1', 'token2', ...]
    '''
    text = text.lower()
    tokens = [word for word in text_to_word_sequence(text) if re.search('[a-zA-Z]', word)]

    # # remove stop words 
    # stop_words = set(stopwords.words('english')) 
    # tokens_remove_stop_words = [w for w in tokens if not w in stop_words]

    # lemmatizer = WordNetLemmatizer() 
    # tokens_lemmatized = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens

def get_data_plain(FILE_NAME, CODE_TO_INDEX):
    data_df = pd.read_csv(FILE_NAME, 
                            sep='\t', 
                            header=None, 
                            names=['code', 'abstract', 'train_or_test'])
    print('before shuffle')
    # print(data_df.head())
    # data_df = data_df.head()
    # data_df = data_df.sample(frac=0.5)
    SAMPLE_SIZE = len(data_df)
    # data_df = shuffle(data_df, random_state=25)
    # data_df = data_df.reset_index(drop=True)
    # print('after shuffle')
    print(data_df.head())
    abstracts = data_df['abstract'].tolist()

    code_index = []
    for i in range(len(data_df)):
        code_index.append(CODE_TO_INDEX[data_df['code'][i]])
    code_index = to_categorical(np.asarray(code_index), num_classes=91)

    embeddings_index = {}
    f = open('./data/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Glove 300d contains total %s word vectors.' % len(embeddings_index))

    data_df['abs_token'] = data_df['abstract'].apply(tokenize_and_remove_stop_words)

    # prepare text samples and their labels
    embedding_matrix = np.zeros((SAMPLE_SIZE, 400, 300))
    papers = []
    for i, content in enumerate(data_df.abs_token.values):
        paper = []
        counter = 0
        for j, v in enumerate(content[:400]):
            embedding_vector = embeddings_index.get(v)
            if embedding_vector is not None:
                embedding_matrix[i, j, :] = embedding_vector
                counter = counter + 1
            paper.append(v)
        papers.append(paper)

    X_train = embedding_matrix
    # X_train=np.expand_dims(embedding_matrix, axis=1),
    y_train = code_index
    return X_train, y_train, papers


def get_data_singlelabel(FILE_NAME, CODE_TO_INDEX, MAX_SEQ_LENGTH = 400, split_index=7983, EMBEDDING_DIM=300):
    data_df = pd.read_csv(FILE_NAME, 
                            sep='\t', 
                            header=None, 
                            names=['code', 'abstract', 'train_or_test'])
    print('before shuffle')
    print(data_df.head())
    data_df = shuffle(data_df, random_state=25)
    data_df = data_df.reset_index(drop=True)
    print('after shuffle')
    print(data_df.head())
    abstracts = data_df['abstract'].tolist()

    code_index = []
    for i in range(len(data_df)):
        code_index.append(CODE_TO_INDEX[data_df['code'][i]])
    code_index = to_categorical(np.asarray(code_index))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(abstracts)
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    data = np.zeros((len(data_df), MAX_SEQ_LENGTH), dtype='int32')

    abs_len_list = []
    papers = []
    for i, abstract in enumerate(abstracts):
        paper = []
        word_tokens = text_to_word_sequence(abstract)
        abs_len_list.append(len(word_tokens))
        j = 0
        for _, word in enumerate(word_tokens):
            if ((word in tokenizer.word_index) 
                    and (j < MAX_SEQ_LENGTH)):
                    # delete maximum number of token.
                    # and (tokenizer.word_index[word] < self.config.data_loader.MAX_NB_WORDS)):

                    data[i, j] = tokenizer.word_index[word]
                    paper.append(word)
                    j = j + 1
        papers.append(paper)

    abs_len_arr = np.array(abs_len_list)
    print("abstract length 0.5 quantile", np.quantile(abs_len_arr, 0.5))

    X_train = data
    y_train = code_index
    # X_train = data[:split_index,:]
    # y_train = code_index[:split_index,:]
    # X_test = data[split_index:,:]
    # y_test = code_index[split_index:,:]
    # print('Shape of X_train tensor:', X_train.shape)
    # print('Shape of y_train tensor:', y_train.shape)
    # print('Shape of X_test tensor:', X_test.shape)
    # print('Shape of y_test tensor:', y_test.shape)

    embeddings_index = {}
    f = open('./data/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Glove 300d contains total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return (X_train, y_train, len(word_index), embedding_matrix, papers)

def set_gpu():
    # set GPU
    # if don't add this, it will report ERROR: Fail to find the dnn implementation.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available")
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
    print(device_lib.list_local_devices(),'\n')