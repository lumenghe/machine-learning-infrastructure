import numpy as np
from sklearn import preprocessing
import pandas as pd

def preprocess(df, config, model=None, mode=None, no_reduction=False):
    """
    Preprocess pipe
    df: input pandas dataframe
    config: config object with field 'preprocessing' (list of str)
    model: model calling preprocessing (can add attributes to the model)
    no_reduction: don't change length of dataframe (modules can be ignored)
    """
    print("Preprocessing [{}]..".format(mode), end="", flush=True)
    if mode == None:
        pplist = config["preprocessing"]
    else:
        pplist = config["preprocessing_{}".format(mode)]
        if pplist is None:
            raise ValueError("missing preprocessing for '{}' (add entry: '{}')".format(mode, "preprocessing_{}".format(mode)))
    if not isinstance(pplist, list):
        pplist = [pplist]
    for pp in pplist:
        if pp == "base":
            df = base(df, config, model)
        elif pp == "fillna":
            df = fillna(df, config)
        elif pp == "target_demean":
            df = target_demean(df, config, model)
        elif pp == "target_sign":
            df = target_sign(df, config, model)
        elif pp == "polynomialfeatures":
            df = polynomialfeatures(df, config)
        elif pp == "target_winsorize":
            df = target_winsorize(df, config)
        elif pp == "target_bound":
            df = target_bound(df, config)
        elif pp == "target_remove_outliers":
            df = target_remove_outliers(df, config, no_reduction=no_reduction)
        else:
            raise ValueError("Unknown preprocessing module '{}'".format(pp))
    print(". Done.", flush=True)
    return df

def base(df, config, model=None):
    print(". base", end="", flush=True)
#    print(df.dtypes)
    for c in df.dtypes[df.dtypes == object].index.values:
        df[c] = (df[c] == True)
#    print(df)
#    assert 0
    return df
