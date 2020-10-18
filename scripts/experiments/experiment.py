from abc import abstractmethod
from typing import Dict
import os

from scripts.utils.helpers import hashlib_hash


class Experiment:
    """
    Class that represents an experiment. It provides utils to identify an experiment using different paths
    and configuration dicts.
    """

    @property
    @abstractmethod
    def path(self):
        """
        Path where the experiment will be written.
        :return:
        """
        pass

    @abstractmethod
    def config(self) -> Dict:
        """
        Configuration to uniquely identify an experiment
        :return:
        """
        pass

    def experiment_id(self) -> str:
        """
        Computes a hash using a config dictionary
        :return:
        """
        return hashlib_hash(sorted(self.config().items()))[:10]

    def _create_directory_if_not_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
