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


def sign_eval(model, config):
    for cat in config["eval_cat"]:
        ytrue = model.get_y(cat=cat)
        if not len(ytrue):
            print("{}: dataset empty.".format(cat))
            continue
        ypred = model.predict(cat=cat)
        ypred_round = ypred.round()
        apr = average_precision_score(ytrue, ypred)
        prec = precision_score(ytrue, ypred_round)
        rec = recall_score(ytrue, ypred_round)
        fpr, tpr, _ = roc_curve(ytrue, ypred)
        roc_auc = auc(fpr, tpr)
        title = "{0}: P = {1:0.2f} ; R = {2:0.2f} ; avgP = {3:0.2f} ; AUC = {4:0.4f}".format(cat, prec, rec, apr, roc_auc)
        print(title)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, alpha=0.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        if config["eval_fig"] is None:
            plt.show()
        else:
            output_fig = config["eval_fig"]
            figname, ext = os.path.splitext(output_fig)
            output_fig = figname + "_" + cat + ext
            fig.savefig(output_fig)
