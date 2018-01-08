import time
from basemodel import BaseModel
from preprocessing import preprocess
from data import *
from sklearn import ensemble
import numpy as np
from sklearn import metrics, datasets
from sklearn import datasets, linear_model
from sklearn.externals import joblib
from sklearn.svm import SVR

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
        if config['model'] == 'sklearn_linear_regression':
            self.model = linear_model.LinearRegression()
        elif config['model'] == 'sklearn_ridge':
            self.model = linear_model.Ridge(alpha=1.0)
        elif config['model'] == 'sklearn_ridgecv':
            self.model = linear_model.RidgeCV()
        elif config['model'] == 'sklearn_lasso':
            self.model = linear_model.Lasso(alpha=1.0)
        elif config['model'] == 'sklearn_lassolars':
            self.model = linear_model.LassoLars(alpha=1.0)
        elif config['model'] == 'sklearn_bayesianridge':
            self.model = linear_model.BayesianRidge()
        elif config['model'] == 'sklearn_randomforest':
            self.model = ensemble.RandomForestRegressor()
        elif config['model'] == 'sklearn_randomforest':
            self.model = ensemble.RandomForestRegressor()
        elif config['model'] == 'sklearn_svr':
            self.model = SVR(kernel='rbf', C=1e3, gamma=0.1)
           # self.model = SVR(kernel='linear', C=1e3)
           # self.model = SVR(kernel='poly', C=1e3, degree=2)

    def init_train_data(self):
        super(Model, self).init_train_data_base()

    def init_submit_data(self, mode):
        super(Model, self).init_submit_data_base(mode)

    def train(self):
        print("Initializing train data...")
        self.init_train_data()
        print("Start model fitting...")
        t = time.time()
        self.model.fit(self.xtrain.values, self.ytrain.values)
        total_time = int(time.time() - t)
        print("Trained model in {} secs".format(total_time))

    def save(self):
        joblib.dump(self.model, self.params)
        print("Saved model at: {}".format(self.params))

    def load(self):
        self.model = joblib.load(self.params)
        print("Loaded model from: {}".format(self.params))
