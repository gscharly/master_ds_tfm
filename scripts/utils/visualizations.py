from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from typing import List, Optional, Dict

import scripts.utils.helpers as helpers
import scripts.conf as conf


def plot_wordcloud(words, width=800, height=800):
    wordcloud = WordCloud(width=width, height=height,
                          background_color='white',
                          min_font_size=10, collocations=False,
                          stopwords=set()).generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def multiple_boxplot(df: pd.DataFrame, num_cols: List, ncols: int, x: Optional[str] = None, width: int = 10,
                     height: int = 10, order: Optional[List] = None):
    fig, axs = plt.subplots(round(len(num_cols)/ncols), ncols, figsize=(width, height))
    y = 0
    for col in num_cols:
        i, j = divmod(y, ncols)
        sns.boxplot(x=x, y=col, data=df, ax=axs[i, j], order=order)
        axs[i, j].xaxis.set_tick_params(rotation=90)
        y += 1
    plt.tight_layout()
    plt.show()


def classification_plots(df: pd.DataFrame, int_class: int):
    """
    Draws two plots:
    - Boxplot comparing scoring distributions for two classes
    - Histogram comparing scoring distributions for two class
    :param df:
    :param int_class:
    :return:
    """
    label = f'label_{int_class}'
    p = f'p{int_class}'

    fig, ax = plt.subplots()
    sns.boxplot(x=label, y=p, data=df)
    ax.set_title(f'Boxplot {p} en función de clase')
    plt.show()

    fig, ax = plt.subplots()
    df_aux_0 = df[df[label] == 0]
    ax.hist(df_aux_0['p0'], bins=20, label='0')
    df_aux_1 = df[df[label] == 1]
    ax.hist(df_aux_1['p0'], bins=20, label='1')
    ax.set_title(f'Histograma {p} en función de clase')
    ax.legend()
    plt.show()


def plot_roc_auc(fpr: np.array, tpr: np.array, roc_auc: float, ths: Optional[np.array] = None,
                 opt_th: Optional[float] = None):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if opt_th and ths is not None:
        nearest_th = helpers.find_nearest(ths, opt_th)
        nearest_th_ix = np.where(ths == nearest_th)
        plt.plot(fpr[nearest_th_ix], tpr[nearest_th_ix], marker='x', markersize=8, color='red',
                 label=f'fpr={fpr[nearest_th_ix][0]:.3f}, tpr={tpr[nearest_th_ix][0]:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.title('ROC AUC')
    plt.show()


def plot_metrics_ths(metrics_dict: Dict, width: int, height: int, th: Optional[float] = None):
    ncols = 2
    nrows = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))
    y = 0
    for metric in conf.CLASS_METRICS_TO_PLOT:
        i, j = divmod(y, ncols)
        metrics_ths = metrics_dict['metrics_ths']
        metrics = metrics_dict[metric]
        axs[i, j].plot(metrics_ths, metrics)
        # Plot max value
        max_value_ix = np.argmax(metrics)
        max_value_th = metrics_ths[max_value_ix]
        # Plot given th
        if th:
            axs[i, j].axvline(x=th, linestyle='--', color='red', label=f'given_th={round(th, 2)}')
        axs[i, j].axvline(x=max_value_th, linestyle='--', color='navy', label=f'opt_th={round(max_value_th, 2)}')
        axs[i, j].set_title(metric)
        axs[i, j].legend()
        y += 1
    plt.show()


def classification_metrics_th(metrics_dict: Dict, th: Optional[float] = None, width: int = 10, height: int = 10):
    # Metric plots
    plot_metrics_ths(metrics_dict, width, height, th=th)
    # ROC AUC plot
    plot_roc_auc(metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['roc_auc'], ths=metrics_dict['roc_ths'],
                 opt_th=th)



