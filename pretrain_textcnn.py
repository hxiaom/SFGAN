import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model


from utils import create_env_dir, get_data_singlelabel, set_gpu

EXP_NAME = 'textcnn'
MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 300
NUM_CLASSES = 91
DROPOUT_RATE = 0.4
NUM_EPOCHS = 10
FILE_NAME = './data/dataset_level2_without_multilabel.txt'
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

create_env_dir(EXP_NAME)

set_gpu()

# Pre train using single discipline proposal
# load NSFC data
print('Load NSFC data.')
X_train, y_train, word_length, embedding_matrix, words = get_data_singlelabel(FILE_NAME, CODE_TO_INDEX)
print('NSFC data loaded.')
print("X_train shape\n", X_train.shape)
print("y_train shape\n", y_train.shape)


# build model
docs_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedding_layer = Embedding(word_length + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False,
                            name='embedding_pretrain'
                            # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                            )
embedded_docs = embedding_layer(docs_input)
kernel_sizes = [3, 4, 5]
pooled = []
for kernel in kernel_sizes:
    conv = Conv1D(filters=100,
                kernel_size=kernel,
                padding='valid',
                strides=1,
                kernel_initializer='he_uniform',
                activation='relu')(embedded_docs)
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
        epochs=NUM_EPOCHS,
        batch_size=64,
        validation_split=0.2,
    )

# save model
textcnn_model.save_weights('./weight_6.h5')
# textcnn_trainer.save()