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

def dump_static(featname, alldf, traindf, work_all, work_train):
    # Generate train features
    feats = pd.DataFrame(work_train.values, index=traindf.index, columns=[featname])
    feats.index.name = traindf.index.name
    write_feats(feats, constant.FEATURE_FACTORY_TRAIN, featname)
    # Generate test features
    feats = pd.DataFrame(work_all.values, index=alldf.index, columns=[featname])
    feats.index.name = alldf.index.name
    for m, path in TESTPATHS:
        write_feats(feats, path, featname)

def create_log_num(featname, alldf, traindf, work_all, work_train):
    # use log
    work_all = np.log(work_all)
    work_train = np.log(work_train)
    dump_static(featname, alldf, traindf, work_all, work_train)


"""
Factories
"""
def past_month_transactions_in_zipcode(featname, traindf, alldf):
    distribution = defaultdict(dict)
    average = dict()
    workdf = traindf.copy()
    workdf["transactionmonth"] = workdf["transactiondate"].apply(lambda d: int(d.strftime("%y%m")))
    for (r, m), s in workdf.groupby(["regionidzip", "transactionmonth"]):
        distribution[r][m] = len(s)
    for r in distribution:
        average[r] = 0
        ct = 0
        for m, v in distribution[r].items():
            if 1601 <= m <= 1609 or 1701 <= m <= 1709:
                ct += 1
                average[r] += v
        average[r] = average[r] / float(ct) if ct else 0.
    # Generate train features
    prevtrans = []
    nanct = 0
    for i, row in workdf[["regionidzip", "transactionmonth"]].iterrows():
        prevm = prev_month(int(row["transactionmonth"]))
        r = row["regionidzip"]
        if math.isnan(r):
            prevtrans.append(np.nan)
            nanct += 1
        else:
            r = int(r)
            prevtrans.append(distribution[r].get(prevm, average[r]))
    print("    NaN for train: {} / {}".format(nanct, len(workdf)))
    feats = pd.DataFrame(prevtrans, index=workdf.index, columns=[featname])
    feats.index.name = workdf.index.name
    write_feats(feats, constant.FEATURE_FACTORY_TRAIN, featname)
    # Generate test features
    nanct = 0
    for m, path in TESTPATHS:
        prevtrans = []
        prevm = prev_month(m)
        for i, r in alldf["regionidzip"].iteritems():
            if math.isnan(r):
                prevtrans.append(np.nan)
                if m == 1610:
                    nanct += 1
            else:
                if r in distribution:
                    prevtrans.append(distribution[r].get(prevm, average[r]))
                else:
                    prevtrans.append(np.nan)
                    if m == 1610:
                        nanct += 1
        if m == 1610:
            print("    NaN for test: {} / {}".format(nanct, len(alldf)))
        feats = pd.DataFrame(prevtrans, index=alldf.index, columns=[featname])
        feats.index.name = alldf.index.name
        write_feats(feats, path, featname)
    return

def past_month_mean_error_in_zipcode(featname, traindf, alldf):
    distribution = defaultdict(dict)
    average = dict()
    workdf = traindf.copy()
    workdf["transactionmonth"] = workdf["transactiondate"].apply(lambda d: int(d.strftime("%y%m")))
    for (r, m), s in workdf.groupby(["regionidzip", "transactionmonth"]):
        distribution[r][m] = s['logerror'].mean()
    for r in distribution:
        average[r] = 0
        ct = 0
        for m, v in distribution[r].items():
            if 1601 <= m <= 1609 or 1701 <= m <= 1709:
                ct += 1
                average[r] += v
        average[r] = average[r] / float(ct) if ct else 0.
    # Generate train features
    prevtrans = []
    nanct = 0
    for i, row in workdf[["regionidzip", "transactionmonth"]].iterrows():
        prevm = prev_month(int(row["transactionmonth"]))
        r = row["regionidzip"]
        if math.isnan(r):
            prevtrans.append(np.nan)
            nanct += 1
        else:
            r = int(r)
            prevtrans.append(distribution[r].get(prevm, average[r]))
    print("    NaN for train: {} / {}".format(nanct, len(workdf)))
    feats = pd.DataFrame(prevtrans, index=workdf.index, columns=[featname])
    feats.index.name = workdf.index.name
    write_feats(feats, constant.FEATURE_FACTORY_TRAIN, featname)
    # Generate test features
    nanct = 0
    for m, path in TESTPATHS:
        prevtrans = []
        prevm = prev_month(m)
        for i, r in alldf["regionidzip"].iteritems():
            if math.isnan(r):
                prevtrans.append(np.nan)
                if m == 1610:
                    nanct += 1
            else:
                if r in distribution:
                    prevtrans.append(distribution[r].get(prevm, average[r]))
                else:
                    prevtrans.append(np.nan)
                    if m == 1610:
                        nanct += 1
        if m == 1610:
            print("    NaN for test: {} / {}".format(nanct, len(alldf)))
        feats = pd.DataFrame(prevtrans, index=alldf.index, columns=[featname])
        feats.index.name = alldf.index.name
        write_feats(feats, path, featname)
    return

def time_from_origin_to_transaction(featname, traindf, alldf):
    from datetime import datetime
    origin_date = datetime(2016,1,1)
    # Generate train features
    time_to_origin = traindf["transactiondate"].apply(lambda t: (t - origin_date).days / 366.)
    feats = pd.DataFrame(time_to_origin.values, index=traindf.index, columns=[featname])
    feats.index.name = traindf.index.name
    write_feats(feats, constant.FEATURE_FACTORY_TRAIN, featname)
    # Generate test features
    month2date = {
            1610: datetime(2016,10,15),
            1611: datetime(2016,11,15),
            1612: datetime(2016,12,15),
            1710: datetime(2017,10,15),
            1711: datetime(2017,11,15),
            1712: datetime(2017,12,15)
        }
    for m, path in TESTPATHS:
        mid_date = month2date[m]
        to_origin = (mid_date - origin_date).days / 366.
        time_to_origin = [to_origin for i in range(len(alldf))]
        feats = pd.DataFrame(time_to_origin, index=alldf.index, columns=[featname])
        feats.index.name = alldf.index.name
        write_feats(feats, path, featname)
    return

def yard_recorded(featname, traindf, alldf):
    work_all = alldf["yardbuildingsqft26"].map(lambda x: 0 if x == "nan" else 1) | alldf["yardbuildingsqft17"].map(lambda x: 0 if math.isnan(x) else 1)
    work_train = traindf["yardbuildingsqft26"].map(lambda x: 0 if x == "nan" else 1) | traindf["yardbuildingsqft17"].map(lambda x: 0 if math.isnan(x) else 1)
    dump_static(featname, alldf, traindf, work_all, work_train)

def log_tax_value(featname, traindf, alldf):
    work_all = pd.to_numeric(alldf["taxvaluedollarcnt"].map(lambda x: np.nan if x == "nan" else x))
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = pd.to_numeric(traindf["taxvaluedollarcnt"].map(lambda x: np.nan if x == "nan" else x)).fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def log_land_tax_value(featname, traindf, alldf):
    work_all = alldf["landtaxvaluedollarcnt"]
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = traindf["landtaxvaluedollarcnt"].fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def log_tax_amount(featname, traindf, alldf):
    work_all = alldf["taxamount"]
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = traindf["taxamount"].fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def log_structure_tax_value(featname, traindf, alldf):
    work_all = alldf["structuretaxvaluedollarcnt"]
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = traindf["structuretaxvaluedollarcnt"].fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def log_lot_size(featname, traindf, alldf):
    work_all = alldf["lotsizesquarefeet"]
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = traindf["lotsizesquarefeet"].fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def log_custom_build_year(featname, traindf, alldf):
    work_all = pd.to_numeric(alldf["yearbuilt"]) - 1770
    median = work_all.median() # for imputation
    work_all = work_all.fillna(median)
    work_train = (pd.to_numeric(traindf["yearbuilt"]) - 1770).fillna(median)
    create_log_num(featname, alldf, traindf, work_all, work_train)

def air_conditioning_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        elif int(x) == 1:
            return 1
        elif int(x) == 13:
            return 2
        else:
            return 3
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "airconditioningtypeid")

def room_count_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        elif int(x) == 0:
            return 1
        elif 1 <= int(x) <= 3:
            return 2
        elif int(x) >= 10:
            return 3
        else: # 4 to 9
            return int(x)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "roomcnt")

def region_county_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        if int(x) == 1286:
            return 1
        elif int(x) == 2061:
            return 2
        elif int(x) == 3101:
            return 3
        return 0
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "regionidcounty")

def property_land_use_cat(featname, traindf, alldf):
    d = {246:1, 247:2, 248:3, 261:4, 266:5, 269:6}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        return d.get(int(x), 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "propertylandusetypeid")

def fips_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if x == "06037":
            return 0
        elif x == "06509":
            return 1
        elif x == "06111":
            return 2
        return 0
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "fips", to_numeric=False)

def bedroom_count_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        if 0 <= int(x) <= 6:
            return int(x)
        elif 7 <= int(x) <= 16:
            return 7
        return 0
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "bedroomcnt")

def bathroom_count_cat(featname, traindf, alldf):
    d = {0:1, 1:2, 1.5:3, 2.0:4, 2.5:5, 3.0:6, 3.5:7, 4:8, 4.5:9, 5:10}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        if 5.5 <= x <= 20:
            return 10
        else:
            return d.get(x, 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "bathroomcnt")

def unit_count_cat(featname, traindf, alldf):
    d = {1:1, 2:2, 3:3, 4:4}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        else:
            return d.get(int(x), 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "unitcnt")

def building_quality_cat(featname, traindf, alldf):
    d = {7:1, 4:2, 1:3, 10:4}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        return d.get(int(x), 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "buildingqualitytypeid")

def garage_count_cat(featname, traindf, alldf):
    d = {0:0, 1:1, 2:2, 3:3}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        if 0 <= x <= 3:
            return x+1
        else:
            return 5
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "garagecarcnt")

def number_stories_cat(featname, traindf, alldf):
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        if 1 <= x <= 2:
            return x
        else:
            return 3
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "numberofstories")

def heating_cat(featname, traindf, alldf):
    d = {2:1, 7:2, 24:3, 6:4}
    def map_to_cat(x):
        if math.isnan(x):
            return 0
        return d.get(int(x), 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "heatingorsystemtypeid")

def property_county_land_use_cat(featname, traindf, alldf):
    d = {"0100":1, "122":2, "010C":3, "0101":4, "34":5, "1111":6, "1":7, "010E":8, "010D":9, "0200":10, "1129":11, "1110":12, "0400":13, "0300":14, "012C":15, "1128":16, "0104":17}
    def map_to_cat(x):
        return d.get(str(x), 0)
    create_one_hot_encoding(featname, traindf, alldf, map_to_cat, "propertycountylandusecode", to_numeric=False)

def sq_binary1(featname, traindf, alldf):
    work_all = alldf["finishedsquarefeet12"].map(lambda x: 0 if math.isnan(x) else 1)
    work_train = traindf["finishedsquarefeet12"].map(lambda x: 0 if math.isnan(x) else 1)
    dump_static(featname, alldf, traindf, work_all, work_train)

def sq_binary2(featname, traindf, alldf):
    work_all = alldf["finishedsquarefeet50"].map(lambda x: 0 if math.isnan(x) else 1)
    work_train = traindf["finishedsquarefeet50"].map(lambda x: 0 if math.isnan(x) else 1)
    dump_static(featname, alldf, traindf, work_all, work_train)

def sq_binary3(featname, traindf, alldf):
    work_all = alldf["finishedsquarefeet15"].map(lambda x: 0 if math.isnan(x) else 1)
    work_train = traindf["finishedsquarefeet15"].map(lambda x: 0 if math.isnan(x) else 1)
    dump_static(featname, alldf, traindf, work_all, work_train)

def sq_binary4(featname, traindf, alldf):
    work_all = alldf["finishedsquarefeet6"].map(lambda x: 0 if math.isnan(x) else 1)
    work_train = traindf["finishedsquarefeet6"].map(lambda x: 0 if math.isnan(x) else 1)
    dump_static(featname, alldf, traindf, work_all, work_train)


"""
Oracles