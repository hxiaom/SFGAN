# TODO: training too slow. Maybe some variables are not in GPU memory or too large.

from data_loader.nsfc_data_loader import NsfcHierDataLoader
from data_loader.functionality_data_loader import FunctionalityDataLoader
from models.nsfc_hier_model import NsfcHierModel
from utils.utils import process_config, create_dirs, get_args
import gc
from tensorflow.python.client import device_lib
import tensorflow as tf
import datetime

def main():
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

    print(device_lib.list_local_devices())

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    # load data
    print('load NSFC data')
    data_loader = NsfcHierDataLoader(config)
    word_index_length, embedding_matrix = data_loader.get_embedding_matrix()
    class_tree = data_loader.get_class_tree()
    max_level = class_tree.get_height()

    print('load sentence functionality data')
    func_data_loader = FunctionalityDataLoader(config)
    X_func, y_func, word_length_func, embedding_matrix_func = func_data_loader.get_train_data()

    # train functionality model
    nsfc_hier_model = NsfcHierModel(config)
    nsfc_hier_model.train_func_classification_model(X_func, y_func, word_length_func, embedding_matrix_func)

    # train each level
    for level in range(max_level):

        # train local classifier
        print("\n### Phase 1: train local classifier ###")
        parents = class_tree.find_at_level(level)
        for parent in parents:
            nsfc_hier_model.instantiate(class_tree=parent, word_index_length=word_index_length, embedding_matrix=embedding_matrix)
            # if parent.model is not None:
            #     print(parent.model)
            #     data = data_loader.get_train_data_by_code(parent.name)
            #     nsfc_hier_model.pretrain(data=data, model=parent.model)

        # train global classifier
        print("\n### Phase 2: train global classifier ###")
        gc.collect()
        global_classifier = nsfc_hier_model.ensemble_classifier(level, class_tree)
        gc.collect()
        if global_classifier == None:
            print('Global classifier is None')
        else:
            print(global_classifier.summary())

        nsfc_hier_model.model.append(global_classifier)
        print('compile')
        nsfc_hier_model.compile(level)
        print('load data')
        level_data = data_loader.get_train_data_by_level(level)
        print('fit', datetime.datetime.now())
        y_pred = nsfc_hier_model.fit(data=level_data, level=level)
        time_iter = datetime.datetime.now()
        print('finish iteration', datetime.datetime.now())
        show_memory()

    print('finish program', datetime.datetime.now())

def show_memory(unit='KB', threshold=1):
    '''查看变量占用内存情况

    :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
    :param threshold: 仅显示内存数值大于等于threshold的变量
    '''
    from sys import getsizeof
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    for i in list(globals().keys()):
        # memory = eval("getsizeof({})".format(i)) // scale
        memory = getsizeof(i)
        if memory >= threshold:
            print(i, memory)

if __name__ == '__main__':
    main()
    print('finish!!!', datetime.datetime.now())