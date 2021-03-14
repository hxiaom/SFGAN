from base.base_trainer import BaseTrain
import os
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.utils import class_weight
import numpy as np

class SLSTMModelTrainer(BaseTrain):
    def __init__(self, model, data_train, data_test, config):
        super(SLSTMModelTrainer, self).__init__(model, data_train, data_test, config)
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

        # self.callbacks.append(
        #     TensorBoard(
        #         log_dir=self.config.callbacks.tensorboard_log_dir,  # not setting
        #         # write_graph=self.config.callbacks.tensorboard_write_graph,
        #     )
        # )

    def train(self):
        # y_int = [y.argmax() for y in self.data_train[1]]
        # class_weights = class_weight.compute_class_weight('balanced',
        #                                          np.arange(45),
        #                                          y_int)
        # print(class_weights)
        # class_weights = {i : class_weights[i] for i in range(45)}
        # print(class_weights)
        history = self.model.fit(
            self.data_train[0], self.data_train[1],
            epochs=self.config.trainer.num_epochs,
            # class_weight=class_weights,
            # verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_data = (self.data_test[0], self.data_test[1]),
            # validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['categorical_accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])