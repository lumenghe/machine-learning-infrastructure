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

    def predict_from_x(self, x):
        raise NotImplementedError("This is BaseModel class")

    def predict(self, cat):
        x = self.get_x(cat)
        return self.predict_from_x(x)

    def get_x(self, cat):
        if cat == "train":
            return self.xtrain.values
        elif cat == "valid":
            return self.xvalid.values
        elif cat == "test":
            return self.xtest.values
        elif cat == "submit":
            return self.xsubmit.values
        else:
            raise ValueError("Unknown category '{}'".format(cat))

    def get_y(self, cat):
        if cat == "train":
            return self.ytrain.values
        elif cat == "valid":
            return self.yvalid.values
        elif cat == "test":
            return self.ytest.values
        else:
            raise ValueError("Unknown category '{}'".format(cat))

    def eval(self):
        if self.config["eval_cat"] is not None:
            full_eval(self, self.config)
        if self.config["submit"]:
            self.clean_train_data()
            submit_for_eval(self, self.config)

    def init_train_data_base(self):
        config = self.config
        dataset = merge_data(config=config, labeled_only=True)
        if config["additional_features"] is not None:
            dataset = merge_additional_features(dataset, config, "train")
        if config["data_split_mode"] == "date":
            self.train, self.valid, self.test = split_data_by_date(dataset, config)
        elif config["data_split_mode"] == "random":
            self.train, self.valid, self.test = split_data_random(dataset, config)
        else:
            raise ValueError("unknown data split mode '{}'".format(config["data_split_mode"]))
        self.train = preprocess(self.train, config, self, mode="train")
        self.valid = preprocess(self.valid, config, self, mode="valid", no_reduction=True)
        self.test = preprocess(self.test, config, self, mode="test", no_reduction=True)
        self.xtrain, self.ytrain = get_xy(self.train, config)
        self.xvalid, self.yvalid = get_xy(self.valid, config)
        self.xtest, self.ytest = get_xy(self.test, config)

    def clean_train_data(self):
        del self.xtrain; del self.ytrain
        del self.xvalid; del self.yvalid
        del self.xtest; del self.ytest
        gc.collect()
