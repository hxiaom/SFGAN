from base.base_model import BaseModel

import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate, Masking
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Attention, GlobalAveragePooling1D, BatchNormalization
from keras.models import Model
from tensorflow.keras import regularizers

class TextCNNModel(BaseModel):
    def __init__(self, word_length, embedding_matrix, configs):
        super(TextCNNModel, self).__init__(configs)
        self.num_classes = 96
        self.build_model(word_length, embedding_matrix)

    def build_model(self, word_length, embedding_matrix):
        
        dropout_rate = 0.4
        embedding_layer = Embedding(word_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )
        
        
        EMBEDDING_DIM = 300

        docs_input = Input(shape=(self.config.data_loader.MAX_DOC_LENGTH,), dtype='int32')
        embedded_docs = embedding_layer(docs_input)

        kernel_sizes = [3, 4, 5]
        pooled = []

        for kernel in kernel_sizes:
            conv = Conv1D(filters=100,
                        kernel_size=kernel,
                        padding='valid',
                        strides=1,
                        kernel_initializer='he_uniform',
                        activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4))(embedded_docs)
            pool = MaxPooling1D(pool_size=self.config.data_loader.MAX_DOC_LENGTH - kernel + 1)(conv)
            pooled.append(pool)

        merged = Concatenate(axis=-1)(pooled)
        flatten = Flatten()(merged)
        # drop = Dropout(rate=dropout_rate)(flatten)
        # x_output = Dense(self.num_classes, activation='sigmoid')(drop)
        x_output = Dense(self.num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4))(flatten)

        self.model = Model(inputs=docs_input, outputs=x_output)
        self.model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc', 
                        tf.keras.metrics.Recall(name='recall'), 
                        tf.keras.metrics.Precision(name='precision'),
                        tfa.metrics.F1Score(name='F1_micro', num_classes=self.num_classes, average='micro'),
                        tfa.metrics.F1Score(name='F1_macro', num_classes=self.num_classes, average='macro')])
