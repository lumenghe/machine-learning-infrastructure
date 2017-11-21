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
