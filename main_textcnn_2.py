import sys
import os
import time
import re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

import innvestigate
import innvestigate.utils as iutils
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from matplotlib.font_manager import FontProperties


EXP_NAME = 'textcnn'
MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 300
NUM_CLASSES = 91
DROPOUT_RATE = 0.4
FILE_NAME = './data/multilabel.txt'
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

def plot_text_heatmap(words, scores, title="", width=15, height=7, verbose=0, max_word_per_line=20):
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
    loc_y = 0.9

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
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

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

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

def get_data_plain():
    data_df = pd.read_csv(FILE_NAME, 
                            sep='\t', 
                            header=None, 
                            names=['code', 'abstract', 'train_or_test'])
    print('before shuffle')
    print(data_df.head())
    data_df = data_df.sample(frac=0.2)
    SAMPLE_SIZE = len(data_df)
    data_df = shuffle(data_df, random_state=25)
    data_df = data_df.reset_index(drop=True)
    print('after shuffle')
    print(data_df.head())
    abstracts = data_df['abstract'].tolist()

    code_index = []
    for i in range(len(data_df)):
        code_index.append(CODE_TO_INDEX[data_df['code'][i]])
    code_index = to_categorical(np.asarray(code_index))

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

# create the experiments dirs
log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), EXP_NAME, "logs/")
checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), EXP_NAME, "checkpoints/")
create_dirs([log_dir, checkpoint_dir])


# set logs
sys.stdout = Logger(f'{log_dir}/output.log', sys.stdout)
sys.stderr = Logger(f'{log_dir}/error.log', sys.stderr)

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

# load NSFC data
print('Load NSFC data')

X_train, y_train, words = get_data_plain()
print("X_train\n", X_train)
print("y_train\n", y_train)


# build model
docs_input = Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM))
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
drop = Dropout(rate=DROPOUT_RATE)(flatten)
x_output = Dense(NUM_CLASSES, 
                kernel_initializer='he_uniform', 
                activation='sigmoid', 
                kernel_regularizer=tf.keras.regularizers.l1(0.01))(drop)

textcnn_model = Model(inputs=docs_input, outputs=x_output)
print(textcnn_model.summary())

# train model
textcnn_model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc', 
                tf.keras.metrics.Recall(name='recall'), 
                tf.keras.metrics.Precision(name='precision')])
history = textcnn_model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=64,
        validation_split=0.2,
    )

# save model
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
a = analyzer.analyze(x, neuron_selection=4)
a = a['input_1']
a = np.squeeze(a)
a = np.sum(a, axis=1)
print(a)
print(a.shape)

print('bbbbbbbbbbbbbbbbbbbbbbbbb')
# words = ['test']*400
plot_text_heatmap(words[0], a.reshape(-1), title='Method: %s' % 'lrp', verbose=0)
plt.savefig('./plot1.png', format='png')
plt.show()