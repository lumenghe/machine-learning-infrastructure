from __future__ import print_function
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from data import read_train


def gaussian(x, mu, sig):
    return ( np.exp(- 0.5 * ((x - mu) / sig)**2) / (np.sqrt(2 * np.pi) * sig) )


def print_info(plot=False):
    df = read_train()
    color = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
    dates = [pd.Timestamp(d) for d in ['2016-01-01', '2016-01-15', '2016-02-01', '2016-02-15', '2016-03-01', '2016-03-15',
    '2016-04-01', '2016-04-15', '2016-05-01', '2016-05-15', '2016-06-01', '2016-06-15', '2016-07-01', '2016-07-15',
    '2016-08-01', '2016-08-15', '2016-09-01', '2016-09-15', '2016-10-01', '2016-10-15', '2016-11-01', '2016-11-15',
    '2016-12-01', '2016-12-15', '2017-01-01']]

    for i, (begin, end) in enumerate(zip(dates, dates[2:])):
      #  sub_df = df.loc[df.apply(lambda row: row['transactiondate'].month, axis=1) == i]
        get_period_info(df, begin, end, color[i%8], plot)

