import time
import xgboost as xgb
from basemodel import BaseModel
from preprocessing import preprocess
from data import *
from eval import sign_eval

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model = None
        self.settings = {}
        self.settings["eta"] = config["eta"]
        self.settings["objective"] = config["objective"]
        self.settings["eval_metric"] = config["eval_metric"]
        self.settings["max_depth"] = 4
        self.settings["silent"] = 1
        self.num_round = config["num_round"]
        self.early_stopping_rounds = config["early_stopping_rounds"]

    def init_train_data(self):
        super(Model, self).init_train_data_base()

    def train(self):
        print("Initializing train data...")
        self.init_train_data()
        print("Start model fitting...")
        t = time.time()
        xgb_train = xgb.DMatrix(self.xtrain.values, label=self.ytrain.values)
        xgb_valid = xgb.DMatrix(self.xvalid.values, label=self.yvalid.values)
        watchlist = [(xgb_train, "train"), (xgb_valid, "valid")]
        self.model = xgb.train(self.settings, xgb_train, self.num_round, watchlist, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=10)
        total_time = int(time.time() - t)
        print("Trained model in {} secs".format(total_time))
