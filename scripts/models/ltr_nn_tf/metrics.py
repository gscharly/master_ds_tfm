# Scripts imports
from scripts.models.metrics_experiment import MetricsExperiment
from scripts.models.ltr_nn_tf.train import LTRNNTFTrain
from scripts.utils.batch_generator import BatchGenerator

# DS
import numpy as np
from scipy.sparse.csr import csr_matrix

# Vis
import matplotlib.pyplot as plt

# Other scripts
import os
import pickle
from typing import Dict, Tuple


class LTRNNTFMetrics(MetricsExperiment):

    def __init__(self, train_exp: LTRNNTFTrain):
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
        batch_gen = BatchGenerator(X, y_true, batch_size=128)
        score_list = model.evaluate(batch_gen, verbose=0)
        return {model.metrics_names[i]: score_list[i] for i in range(len(score_list))}

    def get_metrics(self, data_type: str) -> Dict:
        path_2_read = self.METRICS_PATH[data_type]
        print('Reading metrics from', path_2_read)
        if os.path.exists(path_2_read):
            return pickle.load(open(path_2_read, 'rb'))
        else:
            raise ValueError(f"{path_2_read} does not exist; please execute metrics")

    def show_metrics(self):
        history = self.train_exp.read_model_info()
        for metric, values in history.items():
            plt.plot(values)
            plt.title(metric)
            plt.xlabel('Epochs')
            plt.show()
