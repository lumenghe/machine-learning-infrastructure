import time
import pandas as dp
import numpy as np
from basemodel import BaseModel
from data import *
from preprocessing import preprocess


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model = None
        self.baseline = config["baseline"]

    def init_train_data(self):
        super(Model, self).init_train_data_base()

    def init_submit_data(self, mode):
        super(Model, self).init_submit_data_base(mode)

    def train(self):
        print("Initializing train data...")
        self.init_train_data()
        print("Computing baseline...")
        t = time.time()
        if self.baseline == "mean":
            self.mean = self.train["logerror"].mean()
        elif self.baseline == "mean_city":
            self.mean = self.train["logerror"].mean()
            self.mean_city = dict()
            for city, df in self.train.groupby("regionidcity"):
                self.mean_city[city] = df["logerror"].mean()
        else:
            raise ValueError("Unknown baseline '{}'".format(self.baseline))
        total_time = int(time.time() - t)
        print("Done in {} secs".format(total_time))

    def save(self):
        return
#        with open (self.params, 'w') as f:
#            f.write(str(self.model))
#        print("Saved model at: {}".format(self.params))
