import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, average_precision_score, roc_curve, auc,recall_score,precision_score
import matplotlib.pyplot as plt
from stats import plot_error_distribution
import constant


def full_eval(model, config):
    for cat in config["eval_cat"]:
        ytrue = model.get_y(cat=cat)
        if not len(ytrue):
            print("{}: dataset empty.".format(cat))
            continue
        ypred = model.predict(cat=cat)
        score = mean_absolute_error(ytrue, ypred)
        mean_pred = ypred.mean()
        mean_true = ytrue.mean()
        median_pred = np.median(ypred)
        median_true = np.median(ytrue)
        std_pred = ypred.std()
        std_true = ytrue.std()
        title = "{0}: MAE = {1:.7f}\n pred mean|median|std = {2:.4f}|{3:.4f}|{4:.4f}\n true mean|median|std = {5:.4f}|{6:.4f}|{7:.4f}".format(cat, score, mean_pred, median_pred, std_pred, mean_true, median_true, std_true)
        print(title)
        figname = config["eval_fig"]
        fig, ext = os.path.splitext(figname)
        figname = fig + "_" + cat + ext
        plot_error_distribution(ypred, ytrue, title=title, output_fig=figname)
    return

