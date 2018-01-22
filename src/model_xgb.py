import time
import xgboost as xgb
import scipy.stats as st
from sklearn.model_selection import ParameterGrid, ParameterSampler
from basemodel import BaseModel


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model = None
        self.training_mode = config["training_mode"]
        if self.training_mode == "base":
            self.settings = {
                    "objective": config["objective"],
                    "eval_metric": config["eval_metric"],
                    "eta": config["eta"],
                    "max_depth": 4,
                    "silent": 1,
                }
            self.num_round = config["num_round"]
            self.early_stopping_rounds = config["early_stopping_rounds"]
        elif self.training_mode == "grid":
            self.settings = {
                    "objective": config["objective"],
                    "booster": config["booster"],
                    "eval_metric": config["eval_metric"],
                    "learning_rate": config["learning_rate"],
                    "lambda": config["lambda"],
                    "max_depth": config["max_depth"],
                    "silent": [1],
                }
            self.eval_metric = config["eval_metric"]
            self.n_iter_search = config["n_iter_search"]
            self.random_seed = config["random_seed"]
            self.kfold = config["kfold"]
            self.num_round = config["num_round"]
            self.early_stopping_rounds = config["early_stopping_rounds"]
        elif self.training_mode == "random":
            raise NotImplementedError
        else:
            raise ValueError("unknown training mode '{}'".format(self.training_mode))

    def init_train_data(self):
        super(Model, self).init_train_data_base()

    def init_submit_data(self, mode):
        super(Model, self).init_submit_data_base(mode)

    def train(self):
        print("Initializing train data...")
        self.init_train_data()
        if self.training_mode == "base":
            self.train_base()
        elif self.training_mode == "grid":
            self.train_grid()
        elif self.training_mode == "random":
            self.train_random()
        else:
            raise ValueError("unknown training mode '{}'".format(self.training_mode))

    def train_base(self):
        print("Start model fitting...")
        t = time.time()
        xgb_train = xgb.DMatrix(self.xtrain.values, label=self.ytrain.values)
        xgb_valid = xgb.DMatrix(self.xvalid.values, label=self.yvalid.values)
        watchlist = [(xgb_train, "train"), (xgb_valid, "valid")]
        self.model = xgb.train(self.settings, xgb_train, self.num_round, watchlist, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=10)
        total_time = int(time.time() - t)
        print("Trained model in {} secs".format(total_time))

    def train_grid(self):
        print("Start model fitting in cross validation (grid mode)...")
        t = time.time()
        xgb_train = xgb.DMatrix(self.xtrain.values, label=self.ytrain.values)
        xgb_valid = xgb.DMatrix(self.xvalid.values, label=self.yvalid.values)
        watchlist = [(xgb_train, "train"), (xgb_valid, "valid")]
        current_best = None
        param_grid = ParameterGrid(self.settings)
        for i, params in enumerate(param_grid):
            print("Testing the following params ({}/{}): {}".format(i+1, len(param_grid), params))
            eval_res = dict()
            booster = xgb.train(params, xgb_train, self.num_round, watchlist, early_stopping_rounds=self.early_stopping_rounds, evals_result=eval_res, verbose_eval=10)
            loss = eval_res["valid"]["mae"][booster.best_iteration]
            if current_best is None or loss < current_best:
                current_best = loss
                self.model = booster
        total_time = int(time.time() - t)
        print("Trained model in {} secs".format(total_time))

    def train_random(self):
        print("Start model fitting in cross validation (random mode)...")
        #TODO
        raise NotImplementedError
        print("Trained model in {} secs".format(total_time))
