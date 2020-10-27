# Scripts imports
from scripts.models.metrics_experiment import MetricsExperiment
from scripts.models.ltr_svm_tf.train import LTRSVMTFTrain
import scripts.utils.ml_utils as ml_utils
import scripts.utils.visualizations as vis_utils

# DS
import numpy as np
from scipy.sparse.csr import csr_matrix

# Other scripts
import os
import pickle
from typing import Dict, Tuple


class LTRSVMTFMetrics(MetricsExperiment):

    def __init__(self, train_exp: LTRSVMTFTrain):
        super().__init__()
        self.train_exp = train_exp
        self.METRICS_PATH = {data_type: f'{self.train_exp.path}/{data_type}_metrics.pickle'
                             for data_type in self.DATA_TYPES}

    def _load_data(self, data_type: str) -> Tuple[csr_matrix, np.array]:
        paths = self.train_exp.ltr.datasets_path
        x = pickle.load(open(paths[f'x_{data_type}'], 'rb'))
        y = pickle.load(open(paths[f'y_{data_type}'], 'rb'))
        return x, y

    @property
    def metrics_path(self):
        return self.METRICS_PATH

    def metrics(self, data_type: str) -> Dict:
        model = self.train_exp.read_model()
        X, y_true = self._load_data(data_type)
        y_pred = model.predict(X)
        return ml_utils.regression_metrics(y_true, y_pred)

    def get_metrics(self, data_type: str) -> Dict:
        path_2_read = self.METRICS_PATH[data_type]
        print('Reading metrics from', path_2_read)
        if os.path.exists(path_2_read):
            return pickle.load(open(path_2_read, 'rb'))
        else:
            raise ValueError(f"{path_2_read} does not exist; please execute metrics")

    def show_metrics(self, data_type: str):
        metrics = self.get_metrics(data_type)
        vis_utils.plot_regression_metrics(metrics)
