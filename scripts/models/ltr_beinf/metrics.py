# Scripts imports
from scripts.models.metrics_experiment import MetricsExperiment
from scripts.models.ltr_beinf.train import LTRBEINFTrain
import scripts.conf as conf

import scripts.utils.ml_utils as ml_utils

# DS imports
import pandas as pd

# Other imports
import pickle
from typing import Dict, Tuple


class LTRBEINFMetrics(MetricsExperiment):
    """
    Class that computes metrics for LTR BEINF models. Note that these models are trained in R; this module will
    load and use the predictions of the model from R
    """
    def __init__(self, train_exp: LTRBEINFTrain):
        super().__init__()
        self.train_exp = train_exp
        self.BEINF_PATHS = self._path_dict('beinf')
        self.TH_METRICS_PATH = self._path_dict('th_metrics')
        self.CLASS_METRICS_PATH = self._path_dict('class_metrics')

    def _path_dict(self, name: str):
        return {data_type: f'{self.train_exp.path}/{data_type}_{name}.pickle' for data_type in self.DATA_TYPES}

    def _persist_class_metrics(self, int_class: int, metrics: Dict, data_type: str):
        path_2_write = self.CLASS_METRICS_PATH[data_type]
        path_2_write = path_2_write.replace('class_metrics', f'class_metrics_{int_class}')
        print('Writing metrics to', path_2_write)
        pickle.dump(metrics, open(path_2_write, 'wb'))

    def _read_class_metrics(self, int_class: int, data_type: str):
        path_2_read = self.CLASS_METRICS_PATH[data_type]
        path_2_read = path_2_read.replace('class_metrics', f'class_metrics_{int_class}')
        return pickle.load(open(path_2_read, 'rb'))

    def _persist_th_metrics(self, int_class: int, metrics: Dict, data_type: str):
        path_2_write = self.TH_METRICS_PATH[data_type]
        path_2_write = path_2_write.replace('th_metrics', f'th_metrics_{int_class}')
        print('Writing metrics to', path_2_write)
        pickle.dump(metrics, open(path_2_write, 'wb'))

    def _read_th_metrics(self, int_class: int, data_type: str):
        path_2_read = self.TH_METRICS_PATH[data_type]
        path_2_read = path_2_read.replace('th_metrics', f'th_metrics_{int_class}')
        return pickle.load(open(path_2_read, 'rb'))

    def _persist_class_model_metrics(self, int_class: int, th_metrics: Dict, class_metrics: Dict, data_type: str):
        self._persist_th_metrics(int_class=int_class, metrics=th_metrics, data_type=data_type)
        self._persist_class_metrics(int_class=int_class, metrics=class_metrics, data_type=data_type)

    def read_class_model_metrics(self, int_class: int, data_type: str) -> Tuple[Dict, Dict]:
        return self._read_th_metrics(int_class, data_type), self._read_class_metrics(int_class, data_type)

    def read_predictions(self, data_type: str) -> pd.DataFrame:
        """
        Load predictions (generated using R) for a type of dataset
        :param data_type:
        :return:
        """
        return pd.DataFrame(pickle.load(open(self.BEINF_PATHS[data_type], 'rb')))

    @staticmethod
    def _add_labels(df: pd.DataFrame, int_class: int) -> pd.DataFrame:
        """
        Add a label to the df.
        :param df:
        :param int_class:
        :return:
        """
        assert int_class in [0, 1], 'int_class must be 0 or 1'
        df[f'label_{int_class}'] = df['score'].apply(lambda score: 1 if score == int_class else 0)
        return df

    def _load_data(self, data_type: str) -> pd.DataFrame:
        if data_type == 'train':
            return self.train_exp.train_data()
        elif data_type == 'validation':
            return self.train_exp.ltr.read_validation()
        else:
            return self.train_exp.ltr.read_test()

    def _label_data(self, data_type: str, int_class: int) -> pd.DataFrame:
        """
        Joins information from scores and predictions, and computes classification metrics
        :param data_type:
        :param int_class:
        :return:
        """
        # Join scores and predictions
        df = self._load_data(data_type)
        predictions_df = self.read_predictions(data_type)
        df_all = df.join(predictions_df)
        # Add label for classification problem
        df_label = self._add_labels(df_all, int_class=int_class)
        label = f'label_{int_class}'
        p = f'p{int_class}'
        df_label = df_label[['score', label, p]]
        n0 = len(df_label[df_label[label] == 0])
        n1 = len(df_label[df_label[label] == 1])
        print('Number of 0s:', n0, n0/len(df_label)*100)
        print('Number of 1s:', n1, n1 / len(df_label) * 100)
        return df_label

    @staticmethod
    def th_class_metrics(df: pd.DataFrame, int_class: int) -> Dict:
        """
        Computes classification metrics for different ths
        :param df:
        :param int_class:
        :return:
        """
        y_true = df[f'label_{int_class}'].values
        y_pred_scores = df[f'p{int_class}'].values
        lim_ths = (0.1, 0.8) if int_class == 0 else (0, 0.01)
        metrics = ml_utils.class_metrics_for_ths(y_true, y_pred_scores, lim_ths)
        return metrics

    @staticmethod
    def class_metrics(df: pd.DataFrame, int_class: int):
        y_true = df[f'label_{int_class}'].values
        y_pred = df['y_pred_class'].values
        metrics = ml_utils.class_metrics(y_true, y_pred)
        return metrics

    @staticmethod
    def _assign_new_score_class(df: pd.DataFrame, int_class: int):
        """
        Creates a new column containing the modified score after evaluating a classification model
        :return:
        """
        df[f'mod_class_{int_class}'] = df['y_pred_class'].apply(lambda y_pred:
                                                                float(int_class) if y_pred == 1 else None)
        return df

    def apply_class_model(self, data_type: str, int_class: int, metric: str) -> pd.Series:
        """
        Returns a pandas Series containing all the rows of the original dataset, indicating whether a value
        has been classified as 0 or 1 using the classification models. Values that are non0 /non1 are informed
        as NaN.
        :param data_type: train, test, validation
        :param int_class: 0 or 1
        :param metric: metric to optimize
        :return:
        """
        if metric not in conf.CLASS_METRICS:
            raise ValueError(f'metric must be one of {conf.CLASS_METRICS}')

        print(f'Filtering non {int_class}s from {data_type} dataset')
        df_label = self._label_data(data_type, int_class)
        # Calculate metrics with different ths
        print('Computing metrics with different ths')
        th_metrics = self.th_class_metrics(df_label, int_class)
        # Choose best th for metric
        best_th_for_metric = ml_utils.select_best_th(th_metrics, metric)
        print(f'Using {best_th_for_metric} as th to optimize {metric}')
        th_metrics['opt_th'] = best_th_for_metric
        # Create class labels based on this th
        print('Creating prediction labels and computing classification metrics')
        df_class_label = ml_utils.label_df_with_th(df_label, th=best_th_for_metric, score_col=f'p{int_class}')
        class_metrics = self.class_metrics(df_class_label, int_class)
        # Persist classification metrics
        self._persist_class_model_metrics(int_class=int_class, th_metrics=th_metrics, class_metrics=class_metrics,
                                          data_type=data_type)
        print('Converting df scores')
        df_mod = self._assign_new_score_class(df_class_label, int_class)
        return df_mod[f'mod_class_{int_class}']


