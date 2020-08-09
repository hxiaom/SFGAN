from data_loader.nsfc_data_loader import NsfcHierDataLoader
from data_loader.functionality_data_loader import FunctionalityDataLoader
from models.wstc_model import NsfcHierModel
from trainers.fgan_trainer import FganModelTrainer
from utils.utils import process_config, create_dirs, get_args

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

    # load data
    print('load NSFC data')
    data_loader = NsfcHierDataLoader(config)
    class_tree = data_loader.get_class_tree()
    max_level = class_tree.get_height()

    print('load functionality data')
    func_data_loader = FunctionalityDataLoader(config)
    X_func, y_func, word_length_func, embedding_matrix_func = func_data_loader.get_train_data()

    # train functionality model
    wstc = NsfcHierModel(config, class_tree)
    wstc.train_func(X_func, y_func, word_length_func, embedding_matrix_func)
    word_index_length, embedding_matrix = data_loader.get_embedding_matrix()

    # train each level
    for level in range(max_level):

        # train local classifier
        print("\n### Phase 1: train local classifier ###")
        parents = class_tree.find_at_level(level)
        for parent in parents:
            wstc.instantiate(class_tree=parent, word_index_length=word_index_length, embedding_matrix=embedding_matrix)
            if parent.model is not None:
                print(parent.model)
                data = data_loader.get_train_data_by_code(parent.name)
                wstc.pretrain(data=data, model=parent.model)

        # train global classifier
        print("\n### Phase 2: self-training ###")
        global_classifier = wstc.ensemble_classifier(level, class_tree)
        if global_classifier == None:
            print('Global classifier is NONE')
        else:
            print(global_classifier.summary())

        wstc.model.append(global_classifier)
        wstc.compile(level, loss='kld')
        level_data = data_loader.get_train_data_by_level(level)
        y_pred = wstc.fit(data=level_data, level=level)

if __name__ == '__main__':
    main()