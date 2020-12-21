from base.base_model import BaseModel

import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding, Lambda, Multiply, Concatenate, Masking
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Attention, GlobalAveragePooling1D, BatchNormalization
from keras.models import Model

class FuncModel(BaseModel):
    def __init__(self, word_length, embedding_matrix, configs):
        super(FuncModel, self).__init__(configs)
        self.build_model(word_length, embedding_matrix)

    def build_model(self, word_length, embedding_matrix):
        embedding_layer = Embedding(word_length + 1,
                                    self.config.data_loader.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.config.data_loader.MAX_SENT_LENGTH,
                                    trainable=False
                                    # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                                    )
        # embedding_layer = Masking(mask_value=0)(embedding_layer)
        sentence_input = Input(shape=(self.config.data_loader.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        flat = Flatten()(lstm)
        dense = Dense(50, activation='relu')(flat)
        preds = Dense(5, activation='softmax')(dense)
        self.model = Model(sentence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])

    def load_model(self):
        self.model.load_weights('./experiments/functionality.h5')