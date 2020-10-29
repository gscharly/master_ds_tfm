from abc import abstractmethod
from typing import Dict
import os
import pickle

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

    @property
    def config_path(self) -> str:
        """
        Path to save the configuration
        :return:
        """
        return '{}/config.pickle'.format(self.path)

    def _create_directory_if_not_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _write_config(self):
        self._create_directory_if_not_exists()
        if not os.path.exists(self.config_path):
            print('Writing config in {}'.format(self.config_path))
            with open(self.config_path, 'wb') as fp:
                pickle.dump(self.config(), fp)
