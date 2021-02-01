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
experiment.add_tag('svm')

from data_loader.nsfc_data_loader import NsfcDataLoader


from utils.utils import process_config, create_dirs, get_args
from utils.utils import Logger

from tensorflow.python.client import device_lib
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig

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


    max_len = 384
    configuration = BertConfig()  # default parameters and configuration for BERT

    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "./data/scibert/scibert_scivocab_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer("./data/scibert/scibert_scivocab_uncased/vocab.txt", lowercase=True)

    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    train_path = keras.utils.get_file("train.json", train_data_url)
    eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    eval_path = keras.utils.get_file("eval.json", eval_data_url)

    class SquadExample:
        def __init__(self, question, context, start_char_idx, answer_text, all_answers):
            self.question = question
            self.context = context
            self.start_char_idx = start_char_idx
            self.answer_text = answer_text
            self.all_answers = all_answers
            self.skip = False

        def preprocess(self):
            context = self.context
            question = self.question
            answer_text = self.answer_text
            start_char_idx = self.start_char_idx

            # Clean context, answer and question
            context = " ".join(str(context).split())
            question = " ".join(str(question).split())
            answer = " ".join(str(answer_text).split())

            # Find end character index of answer in context
            end_char_idx = start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return

            # Mark the character indexes in context that are in answer
            is_char_in_ans = [0] * len(context)
            for idx in range(start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1

            # Tokenize context
            tokenized_context = tokenizer.encode(context)

            # Find tokens that were created from answer characters
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)

            if len(ans_token_idx) == 0:
                self.skip = True
                return

            # Find start and end token index for tokens from answer
            start_token_idx = ans_token_idx[0]
            end_token_idx = ans_token_idx[-1]

            # Tokenize question
            tokenized_question = tokenizer.encode(question)

            # Create inputs
            input_ids = tokenized_context.ids + tokenized_question.ids[1:]
            token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
                tokenized_question.ids[1:]
            )
            attention_mask = [1] * len(input_ids)

            # Pad and create attention masks.
            # Skip if truncation is needed
            padding_length = max_len - len(input_ids)
            if padding_length > 0:  # pad
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
            elif padding_length < 0:  # skip
                self.skip = True
                return

            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_mask = attention_mask
            self.start_token_idx = start_token_idx
            self.end_token_idx = end_token_idx
            self.context_token_to_char = tokenized_context.offsets


    with open(train_path) as f:
        raw_train_data = json.load(f)

    with open(eval_path) as f:
        raw_eval_data = json.load(f)


    def create_squad_examples(raw_data):
        squad_examples = []
        for item in raw_data["data"]:
            for para in item["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    question = qa["question"]
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    squad_eg = SquadExample(
                        question, context, start_char_idx, answer_text, all_answers
                    )
                    squad_eg.preprocess()
                    squad_examples.append(squad_eg)
        return squad_examples


    def create_inputs_targets(squad_examples):
        dataset_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "start_token_idx": [],
            "end_token_idx": [],
        }
        for item in squad_examples:
            if item.skip == False:
                for key in dataset_dict:
                    dataset_dict[key].append(getattr(item, key))
        for key in dataset_dict:
            dataset_dict[key] = np.array(dataset_dict[key])

        x = [
            dataset_dict["input_ids"],
            dataset_dict["token_type_ids"],
            dataset_dict["attention_mask"],
        ]
        y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
        return x, y

    def create_model():
        ## BERT encoder
        encoder = TFBertModel.from_pretrained("bert-base-uncased")

        ## QA Model
        input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
        embedding = encoder(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]

        start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
        start_logits = layers.Flatten()(start_logits)

        end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
        end_logits = layers.Flatten()(end_logits)

        start_probs = layers.Activation(keras.activations.softmax)(start_logits)
        end_probs = layers.Activation(keras.activations.softmax)(end_logits)

        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[start_probs, end_probs],
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam(lr=5e-5)
        model.compile(optimizer=optimizer, loss=[loss, loss])
        return model

    use_tpu = True
    if use_tpu:
        # Create distribution strategy
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        # Create model
        with strategy.scope():
            model = create_model()
    else:
        model = create_model()

    model.summary()

    train_squad_examples = create_squad_examples(raw_train_data)
    x_train, y_train = create_inputs_targets(train_squad_examples)
    print(f"{len(train_squad_examples)} training points created.")

    eval_squad_examples = create_squad_examples(raw_eval_data)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples)
    print(f"{len(eval_squad_examples)} evaluation points created.")


    # load NSFC data
    print('Load NSFC data')
    data_loader = NsfcDataLoader(config)
    X_train, y_train, X_test, y_test = data_loader.get_train_data_tfidf()
    print("X_train\n", X_train)
    print("y_train\n", y_train)

    svc = SVC(kernel = 'linear') 
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()