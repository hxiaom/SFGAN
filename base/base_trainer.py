class BaseTrain(object):
    def __init__(self, model, data_train, data_test, config):
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.config = config

    def train(self):
        raise NotImplementedError