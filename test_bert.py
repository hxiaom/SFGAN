# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = TFBertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)

from transformers import BertTokenizer, TFBertModel
from transformers import glue_convert_examples_to_features
import tensorflow_datasets
from transformers import glue_convert_examples_to_features



tokenizer = BertTokenizer.from_pretrained("/home/hxiaom/workstation/SFGAN/data/scibert_scivocab_uncased/")

model = TFBertModel.from_pretrained("/home/hxiaom/workstation/SFGAN/data/scibert_scivocab_uncased/", from_pt=True)

input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)

embedding_layer = Embedding(word_length + 1,
                            self.config.data_loader.EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=self.config.data_loader.MAX_SENT_LENGTH,
                            trainable=False
                            # mask_zero=True  # mask will report ERROR: CUDNN_STATUS_BAD_PARAM
                            )
# embedding_layer = Masking(mask_value=0)(embedding_layer)
sentence_input = Input(shape=(50,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
flat = Flatten()(lstm)
dense = Dense(200, activation='relu')(flat)
preds = Dense(5, activation='softmax')(dense)