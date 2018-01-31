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


if __name__  == "__main__":
    configs = sys.argv[1:]
    for cfg in configs:
        t = time.time()
        try:
            print("#" * 80)
            print("RUNNING {}".format(cfg))
            print("#" * 80)
            config = Config(cfg)
            print(str(config))
            run(config)
        except:
            print("RUN CRASHED!\n########### TRACE ###########".format(cfg))
            traceback.print_exc()
            print("######## END OF TRACE ########".format(cfg))
        s = time.time() - t
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        total_time = "%d:%02d:%02d" % (h, m, s)
        print("\nTOTAL RUNTIME = {}".format(total_time))
