from abc import abstractmethod
from typing import Dict

import pandas as pd


class MetricsExperiment:
    DATA_TYPES = ['train', 'validation', 'test']

    def __init__(self):
        pass

    @abstractmethod
    def metrics_train(self):
        pass

    @abstractmethod
    def metrics_val(self):
        pass

    @abstractmethod
    def metrics_test(self):
        pass

    @abstractmethod
    def get_metrics(self, data_type: str) -> Dict:
        """Reads available metrics, if they exist."""
        pass

    @abstractmethod
    def metrics(self, df: pd.DataFrame) -> Dict:
        """Computes a dictionary with metrics"""
        pass

    def run_metric(self, data_type: str):
        """Calculates and persists metrics for a given dataset type"""
        if data_type not in self.DATA_TYPES:
            raise ValueError("data_type must be one of train, validation, test")
        elif data_type == 'train':
            print('Calculating train metrics...')
            self.metrics_train()
        elif data_type == 'validation':
            print('Calculating validation metrics...')
            self.metrics_val()
        else:
            print('Calculating test metrics...')
            self.metrics_test()

    @abstractmethod
    def persist_metrics(self, metrics: Dict, data_type: str):
        pass

    def run(self):
        """Computes and persists metrics for train, test and validation datasets"""
        for data_type in self.DATA_TYPES:
            self.run_metric(data_type)
