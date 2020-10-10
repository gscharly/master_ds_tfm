import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

from typing import Dict, Tuple

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train_validate_test_split(df: pd.DataFrame, train_percent: float = .6, validate_percent: float = .2,
                              seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def predict_label(y_pred_scores: np.array, th: float) -> np.array:
    return np.array(list(map(lambda score: 1 if score >= th else 0, y_pred_scores)))


def class_metrics_for_ths(y_true: np.array, y_pred_scores: np.array, lim_ths: Tuple) -> Dict:
    """
    Returns a dictionary with multiple classfication metrics depending on different thresholds
    :param y_true: array containing true labels
    :param y_pred_scores: array containing score predictions
    :param lim_ths:
    :return:
    """
    fpr, tpr, ths = roc_curve(y_true, y_pred_scores)
    # TH generation to search for best cutoff
    metrics_th = np.linspace(min(lim_ths), max(lim_ths), 20)
    y_pred_ths = [predict_label(y_pred_scores, th) for th in metrics_th]
    return {
        'accuracy': [accuracy_score(y_true, y_pred) for y_pred in y_pred_ths],
        'precision': [precision_score(y_true, y_pred) for y_pred in y_pred_ths],
        'recall': [recall_score(y_true, y_pred) for y_pred in y_pred_ths],
        'f1': [f1_score(y_true, y_pred) for y_pred in y_pred_ths],
        'roc_auc': roc_auc_score(y_true, y_pred_scores),
        'fpr': fpr,
        'tpr': tpr,
        'roc_ths': ths,
        'metrics_ths': metrics_th
    }


def class_metrics(y_true: np.array, y_pred: np.array) -> Dict:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'conf_matrix': confusion_matrix(y_true, y_pred)
    }


def select_best_th(metrics_dict: Dict, metric: str):
    """
    Select the th that maximizes a metric
    :param metrics_dict:
    :param metric:
    :return:
    """
    max_metric_ix = np.argmax(metrics_dict[metric])
    return metrics_dict['metrics_ths'][max_metric_ix]


def label_df_with_th(df: pd.DataFrame, th: float, score_col: str):
    """
    Creates a prediction for a label, given a th. If score > th -> 1
    :param df:
    :param th:
    :param score_col:
    :return:
    """
    df['y_pred_class'] = df[score_col].apply(lambda score: 1 if score >= th else 0)
    return df
