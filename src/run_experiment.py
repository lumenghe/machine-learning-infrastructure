import sys
import time
import traceback
import gc
from config import Config


def get_model(config):
    name = config["model"]
    if name == "baseline":
        from model_baseline import Model
        return Model(config)
    elif name == "xgb":
        from model_xgb import Model
        return Model(config)
    elif name == "xgb_sign":
        from model_xgb_sign import Model
        return Model(config)
    elif name.startswith('sklearn'):
        from model_sklearn import Model
        return Model(config)
    elif name == "lgb":
        from model_lgb import Model
        return Model(config)
    else:
        raise ValueError("Model '{}' in not registered!".format(name))

def run(config):
    model = get_model(config)
    print ('has_params=', model.has_params())
    if not model.has_params() or config["force_rerun"]:
        model.train()
        model.save()
    else:
        model.load()
    model.eval()
    del model
    gc.collect()
    return

