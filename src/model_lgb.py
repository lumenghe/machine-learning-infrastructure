import time
import lightgbm as lgb
from basemodel import BaseModel


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model = None
        self.settings = {
                "max_bin": config["max_bin"],
                "learning_rate": config["learning_rate"],
                "boosting_type": config["boosting_type"],
                "objective": config["objective"],
                "metric": config["metric"],
                "sub_feature": config["sub_feature"],
                "bagging_fraction": config["bagging_fraction"],
                "bagging_freq": config["bagging_freq"],
                "num_leaves": config["num_leaves"],
                "min_data": config["min_data"],
                "min_hessian": config["min_hessian"],
            }
        self.num_boost_round = config["num_boost_round"]
        self.early_stopping_round = config["early_stopping_round"]

    def init_train_data(self):
        super(Model, self).init_train_data_base()

    def init_submit_data(self, mode):
        super(Model, self).init_submit_data_base(mode)

    def train(self):
        print("Initializing train data...")
        self.init_train_data()
        print("Start model fitting...")
        t = time.time()
        lgb_train = lgb.Dataset(self.xtrain.values, self.ytrain.values.reshape(len(self.ytrain)))
        lgb_valid = lgb.Dataset(self.xvalid.values, self.yvalid.values.reshape(len(self.yvalid)), reference=lgb_train)
        self.model = lgb.train(self.settings, lgb_train, self.num_boost_round, [lgb_valid])
        total_time = int(time.time() - t)
        self.model.reset_parameter({"num_threads":1})
        print("Trained model in {} secs".format(total_time))

    def save(self):
        self.model.save_model(self.params)
        print("Saved model at: {}".format(self.params))

    def load(self):
        self.model = lgb.Booster(model_file=self.params)
        print("Loaded model from: {}".format(self.params))
