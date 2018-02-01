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


def get_period_info(df, start_date, end_date, color, plot=False):
    sub_df = df.loc[df.apply(lambda row: start_date <= row['transactiondate'] < end_date, axis=1)]
    mu = sub_df['logerror'].mean()
    sig = sub_df['logerror'].std()
    histogram_df = pd.DataFrame({'logerror': sub_df['logerror']})

    if plot:
        plt.hist(histogram_df.values, bins='auto', range=(-0.5, 0.5), normed=True)
        x = np.linspace(-0.5, 0.5, num=100)
        plt.plot(x, gaussian(x, mu, sig), color)
        plt.show()
        sm.qqplot(histogram_df.values, line='45', fit=True)
        plt.show()

    vals = histogram_df.values
    quartile_1 = histogram_df.quantile(q=0.25)['logerror']
    median = histogram_df.quantile(q=0.5)['logerror']
    quartile_3 = histogram_df.quantile(q=0.75)['logerror']
    print("[{0} {1}) mean = {2}, std = {3}, quartile_1 = {4}, median = {5}, quartile_1 = {6}, SKEW = {7} KURTOSIS = {8} data len={9}".format(
    start_date, end_date, mu, sig, quartile_1 , median, quartile_3 , skew(vals), kurtosis(vals), len(sub_df.index)))

