from abc import abstractmethod
from typing import Dict


class MetricsExperiment:
    DATA_TYPES = ['train', 'validation', 'test']

    def __init__(self):
        pass

    @abstractmethod
    def get_metrics(self, data_type: str) -> Dict:
        """Reads available metrics, if they exist."""
        pass

    @abstractmethod
    def metrics(self, data_type: str) -> Dict:
        """Computes a dictionary with metrics"""
        pass

    @abstractmethod
    def persist_metrics(self, metrics: Dict, data_type: str):
        pass

    def run_metric(self, data_type: str):
        """Calculates and persists metrics for a given dataset type"""
        if data_type not in self.DATA_TYPES:
            raise ValueError("data_type must be one of train, validation, test")
        else:
            print(f'Computing metric for {data_type} dataset')
            metrics = self.metrics(data_type=data_type)
            self.persist_metrics(metrics, data_type=data_type)

    def run(self):
        """Computes and persists metrics for train, test and validation datasets"""
        for data_type in self.DATA_TYPES:
            self.run_metric(data_type)
