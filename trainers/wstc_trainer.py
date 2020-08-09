
from base.base_trainer import BaseTrain
from keras.callbacks import ModelCheckpoint, TensorBoard


class WstcModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(FganModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        # train local classifier 
        for level in range(max_level):
            y_pred = proceed_level(x, sequences, wstc, args, pretrain_epochs, self_lr, decay, update_interval,
                                delta, class_tree, level, expand_num, background_array, max_doc_length, max_sent_length,
                                len_avg, len_std, beta, alpha, vocabulary_inv, common_words)

        # train global classifierfd

    # def proceed_level(x, sequences, wstc, args, pretrain_epochs, self_lr, decay, update_interval,
    #                 delta, class_tree, level, expand_num, background_array, doc_length, sent_length, len_avg,
    #                 len_std, num_doc, interp_weight, vocabulary_inv, common_words):
    def proceed_level(self, level):
        print(f"\n### Proceeding level {level} ###")
        dataset = args.dataset
        sup_source = args.sup_source
        maxiter = args.maxiter.split(',')
        maxiter = int(maxiter[level])
        batch_size = args.batch_size
        parents = class_tree.find_at_level(level)
        parents_names = [parent.name for parent in parents]
        print(f'Nodes: {parents_names}')
        
        for parent in parents:
            # initialize classifiers in hierarchy
            print("\n### Input preparation ###")

            if class_tree.embedding is None:
                train_class_embedding(x, vocabulary_inv, dataset_name=args.dataset, node=class_tree)
            parent.embedding = class_tree.embedding
            wstc.instantiate(class_tree=parent)
            
            save_dir = f'./results/{dataset}/{sup_source}/level_{level}'

            if parent.model is not None:
                
                num_real_doc = len(seed_docs) / 5

                if sup_source == 'docs':
                    real_seed_docs, real_seed_label = augment(x, parent.children, num_real_doc)
                    print(f'Labeled docs {len(real_seed_docs)} + Pseudo docs {len(seed_docs)}')
                    seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
                    seed_label = np.concatenate((seed_label, real_seed_label), axis=0)

                perm = np.random.permutation(len(seed_label))
                seed_docs = seed_docs[perm]
                seed_label = seed_label[perm]

                print('\n### Phase 2: pre-training with pseudo documents ###')
                print(f'Pretraining node {parent.name}')

                wstc.pretrain(x=seed_docs, pretrain_labels=seed_label, model=parent.model,
                            optimizer=SGD(lr=0.1, momentum=0.9),
                            epochs=pretrain_epochs, batch_size=batch_size,
                            save_dir=save_dir, suffix=parent.name)

        global_classifier = wstc.ensemble_classifier(level)
        wstc.model.append(global_classifier)
        t0 = time()
        print("\n### Phase 3: self-training ###")
        selftrain_optimizer = SGD(lr=self_lr, momentum=0.9, decay=decay)
        wstc.compile(level, optimizer=selftrain_optimizer, loss='kld')
        y_pred = wstc.fit(x, level=level, tol=delta, maxiter=maxiter, batch_size=batch_size,
                        update_interval=update_interval, save_dir=save_dir)
        print(f'Self-training time: {time() - t0:.2f}s')
        return y_pred