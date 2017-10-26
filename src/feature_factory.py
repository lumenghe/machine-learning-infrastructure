import os, sys
import gc
import math
import time
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn import preprocessing
from data import *
import constant


"""
Utils
"""
TESTPATHS = [
    (1610, constant.FEATURE_FACTORY_TEST_1610),
    (1611, constant.FEATURE_FACTORY_TEST_1611),
    (1612, constant.FEATURE_FACTORY_TEST_1612),
    (1710, constant.FEATURE_FACTORY_TEST_1710),
    (1711, constant.FEATURE_FACTORY_TEST_1711),
    (1712, constant.FEATURE_FACTORY_TEST_1712)
]

def prev_month(ym):
    if ym in [1601, 1611, 1612, 1701, 1711, 1712]: # WARNING: ONLY WHEN WE DON'T HAVE ACCES TO 2017 TRAIN DATA
        return None
    else:
        return (ym - 1)

def write_feats(df, path, featname):
    fpath = os.path.join(path, featname + ".pkl")
    df.to_pickle(fpath, constant.FEATURE_FACTORY_COMPRESSION)
    print("    Wrote: {}".format(fpath))

def read_feats(path, featname):
    fpath = os.path.join(path, featname + ".pkl")
    df = pd.read_pickle(fpath, constant.FEATURE_FACTORY_COMPRESSION)
    return df

def create_one_hot_encoding(featname, alldf, traindf, map_to_cat, raw_featname, to_numeric=True):
    if to_numeric:
        work_all = pd.to_numeric(alldf[raw_featname]).apply(map_to_cat)
        work_train = pd.to_numeric(traindf[raw_featname]).apply(map_to_cat)
    else:
        work_all = alldf[raw_featname].apply(map_to_cat)
        work_train = traindf[raw_featname].apply(map_to_cat)
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(work_all.values.reshape(-1,1))
    # Generate train features
    ohe_values = encoder.transform(work_train.values.reshape(-1,1)).toarray()
    ohe_labels = [featname + "_" + str(i) for i in range(ohe_values.shape[1])]
    feats = pd.DataFrame(ohe_values, index=traindf.index, columns=ohe_labels)
    feats.index.name = traindf.index.name
    write_feats(feats, constant.FEATURE_FACTORY_TRAIN, featname)
    # Generate test features
    ohe_values = encoder.transform(work_all.values.reshape(-1,1)).toarray()
    ohe_labels = [featname + "_" + str(i) for i in range(ohe_values.shape[1])]
    feats = pd.DataFrame(ohe_values, index=alldf.index, columns=ohe_labels)
    feats.index.name = alldf.index.name
    for m, path in TESTPATHS:
        write_feats(feats, path, featname)
