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

def fillna(df, config):
    print(". fillna", end="", flush=True)
    v = config["fillna"]
    if v is None:
        raise ValueError("fillna value missing from config")
    df = df.fillna(v)
    return df

def target_demean(df, config, model=None):
    print(". target_demean", end="", flush=True)
    target = config["target"]
    train = df.loc[df.apply(lambda row: config["train_start"] <= row['transactiondate'] < config["train_end"], axis=1)]
    mean = train[target].mean()
    df[target] = df[target].values - mean
    if model is not None:
        model.mean_target = mean
    return df

def target_sign(df, config, model=None):
    print(". target_sign", end="", flush=True)
    target = config["target"]
    df[target] = (df[target].values >= 0).astype(np.int32)
    return df

def polynomialfeatures(df, config):
    print(". polynomialfeatures", end="", flush=True)
    x1, x2 = config['polynomialfeatures']
    input_df = df[config['polynomialfeatures']]
    poly = preprocessing.PolynomialFeatures(3)
    output_nparray = poly.fit_transform(input_df)
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(config['polynomialfeatures'],p) for p in poly.powers_]]
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    df.join(output_df)
    return df

def target_winsorize(df, config):
    print(". target_winsorize", end="", flush=True)
    target = config["target"]
    train = df.loc[df.apply(lambda row: config["train_start"] <= row['transactiondate'] < config["train_end"], axis=1)]
    left_quantile = train.quantile(config["target_winsor_left"])
    right_quantile = train.quantile(config["target_winsor_right"])
    df[target] = df[target].clip(left_quantile, right_quantile)
    return df
