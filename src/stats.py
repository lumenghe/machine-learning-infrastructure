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

