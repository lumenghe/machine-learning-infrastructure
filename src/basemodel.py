import os
import gc
from eval import full_eval, submit_for_eval
from data import *
from preprocessing import preprocess


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.xtrain = None
        self.ytrain = None
        self.xvalid = None
        self.yvalid = None
        self.xtest = None
        self.ytest = None
        self.xsubmit = None
        self.submitindex = None
        self.params = config["params"]
        return

    def has_params(self):
        return os.path.isfile(self.params)

    def init_train_data(self):
        raise NotImplementedError("This is BaseModel class")

    def init_submit_data(self, mode):
        raise NotImplementedError("This is BaseModel class")

    def train(self):
        raise NotImplementedError("This is BaseModel class")

    def save(self):
        raise NotImplementedError("This is BaseModel class")

    def load(self):
        raise NotImplementedError("This is BaseModel class")
