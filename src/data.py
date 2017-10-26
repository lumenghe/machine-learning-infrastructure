import os
import pandas as pd
import numpy as np
import constant
import datetime
from config import Config


def read_properties():
    print("Reading PROPERTIES DATA... ", end="", flush=True)
    if os.path.exists(constant.PROPERTIES_2016_PKL):
        df = pd.read_pickle(constant.PROPERTIES_2016_PKL)
    else:
        df = pd.read_csv(constant.PROPERTIES_2016,
            header = 0,
            index_col = 0,
            parse_dates = [46,51],
            infer_datetime_format = True,
            dtype = { "parcelid": np.int64,
                    "airconditioningtypeid": np.float64,
                    "architecturalstyletypeid": str,
                    "basementsqft": np.float64,
                    "bathroomcnt": np.float64,
                    "bedroomcnt": np.float64,
                    "buildingclasstypeid": str,
                    "buildingqualitytypeid": str,
                    "calculatedbathnbr": np.float64,
                    "decktypeid": str,
                    "finishedfloor1squarefeet": np.float64,
                    "calculatedfinishedsquarefeet": np.float64,
                    "finishedsquarefeet12": np.float64,
                    "finishedsquarefeet13": np.float64,
                    "finishedsquarefeet15": np.float64,
                    "finishedsquarefeet50": np.float64,
                    "finishedsquarefeet6": np.float64,
                    "fips": str,
                    "fireplacecnt": np.float64,
                    "fullbathcnt": np.float64,
                    "garagecarcnt": np.float64,
                    "garagetotalsqft": np.float64,
                    "hashottuborspa": str,
                    "heatingorsystemtypeid": str,
                    "latitude": np.float64,
                    "longitude": np.float64,
                    "lotsizesquarefeet": np.float64,
                    "poolcnt": np.float64,
                    "poolsizesum": np.float64,
                    "pooltypeid10": str,
                    "pooltypeid2": str,
                    "pooltypeid7": str,
                    "propertycountylandusecode": str,
                    "propertylandusetypeid": str,
                    "propertyzoningdesc": str,
                    "rawcensustractandblock": str,
                    "regionidcity": str,
                    "regionidcounty": str,
                    "regionidneighborhood": str,
                    "regionidzip": str,
                    "roomcnt": np.float64,
                    "storytypeid": str,
                    "threequarterbathnbr": np.float64,
                    "typeconstructiontypeid": str,
                    "unitcnt": np.float64,
                    "yardbuildingsqft17": np.float64,
                    "yardbuildingsqft26": np.float64,
                    "yearbuilt": str, # date
                    "numberofstories": np.float64,
                    "fireplaceflag": str,
                    "structuretaxvaluedollarcnt": np.float64,
                    "taxvaluedollarcnt": np.float64,
                    "assessmentyear": str, # date
                    "landtaxvaluedollarcnt": np.float64,
                    "taxamount": np.float64,
                    "taxdelinquencyflag": str,
                    "taxdelinquencyyear": str,
                    "censustractandblock": str}
            )
        df.to_pickle(constant.PROPERTIES_2016_PKL)
        print("Created pickle file: {}".format(constant.PROPERTIES_2016_PKL))
    _df_properties = df
    return df.copy()

def read_train(drop_duplicates=True):
    print("Reading TRAIN DATA... ", end="", flush=True)
    if os.path.exists(constant.TRAIN_2016_PKL):
        df = pd.read_pickle(constant.TRAIN_2016_PKL)
    else:
        df = pd.read_csv(constant.TRAIN_2016,
            header = 0,
            index_col = 0,
            parse_dates = [2],
            infer_datetime_format = True,
            )
        df.to_pickle(constant.TRAIN_2016_PKL)
        print("Created pickle file: {}".format(constant.TRAIN_2016_PKL))
    df = df.reset_index().drop_duplicates(subset="parcelid", keep="last").set_index("parcelid")
    _df_train = df
    return df.copy()

def split_data_by_date(df, config):
    print("Splitting Data by date... ", end="", flush=True)
    train = df.loc[df.apply(lambda row: config["train_start"] <= row['transactiondate'] < config["train_end"], axis=1)]
    valid = df.loc[df.apply(lambda row: config["valid_start"] <= row['transactiondate'] < config["valid_end"], axis=1)]
    test = df.loc[df.apply(lambda row: config["test_start"] <= row['transactiondate'] < config["test_end"], axis=1)]
    print("Done.", flush=True)
    return train, valid, test

def split_data_random(df, config):
    print("Splitting Data at random... ", end="", flush=True)
    df = df.sample(frac=1, random_state=config["random_seed"])
    length = len(df)
    split1 = int(length * config["random_split_ratio"][0])
    split2 = int(length * config["random_split_ratio"][1])
    train = df.iloc[:split1]
    valid = df.iloc[split1:split2]
    test = df.iloc[split2:]
    print("Done.", flush=True)
    return train, valid, test
