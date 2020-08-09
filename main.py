from data_loader.nsfc_data_loader import NsfcHierDataLoader
from data_loader.functionality_data_loader import FunctionalityDataLoader
from models.wstc_model import NsfcHierModel
from trainers.fgan_trainer import FganModelTrainer
from utils.utils import process_config, create_dirs, get_args
from time import time


def main():
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

    print('Load data')
    data_loader = NsfcHierDataLoader(config)
    class_tree = data_loader.get_class_tree()
    max_level = class_tree.get_height()

    wstc = NsfcHierModel(config, class_tree)

    func_data_loader = FunctionalityDataLoader(config)
    X_func, y_func, word_length_func, embedding_matrix_func = func_data_loader.get_train_data()
    wstc.train_func(X_func, y_func, word_length_func, embedding_matrix_func)

    word_index_length, embedding_matrix = data_loader.get_embedding_matrix()
    

    # train each level
    models = {}
    for level in range(max_level):
        parents = class_tree.find_at_level(level)
        parents_names = [parent.name for parent in parents]

        # train each node
        for parent in parents:
            wstc.instantiate(class_tree=parent, word_index_length=word_index_length, embedding_matrix=embedding_matrix)
            
            if parent.model is not None:
                print(parent.model)
                data = data_loader.get_train_data_by_code(parent.name)
                wstc.pretrain(data=data, model=parent.model)

        global_classifier = wstc.ensemble_classifier(level, class_tree)
        if global_classifier == None:
            print('Global classifier NONE')
        else:
            print(global_classifier.summary())
        wstc.model.append(global_classifier)
        t0 = time()
        print("\n### Phase 3: self-training ###")
        # selftrain_optimizer = SGD(lr=self_lr, momentum=0.9, decay=decay)
        wstc.compile(level, loss='kld')
        level_data = data_loader.get_train_data_by_level(level)
        y_pred = wstc.fit(data=level_data, level=level)
        print(f'Self-training time: {time() - t0:.2f}s')


if __name__ == '__main__':
    main()