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
