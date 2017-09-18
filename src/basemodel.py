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
