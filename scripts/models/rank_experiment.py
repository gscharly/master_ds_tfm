# Scripts imports
from scripts.experiments.experiment import Experiment
from scripts.models.ltr_beinf.train import LTRBEINFTrain
from scripts.extractive_summary.ltr.ltr_features_targets import LTRFeaturesTargets
from scripts.extractive_summary.ltr.ltr_features_targets_tf import LTRFeaturesTargetsTF
from scripts.conf import MODELS_PATH

# DS imports
import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

# Other
from typing import Dict, Tuple, Union
from abc import abstractmethod, ABC
import os


class RankExperiment(Experiment, ABC):
    """
    Class that represents a ranking system. Every class that extends this one must provide a ranking method, that
    will use a score assigned to each event for a given match.
    Important: each match must be ranked separately!
    """

    def __init__(self, ltr, n: int):
        """
        This class will need a model that assigns the scores to each event, and a number of events to output.
        :param ltr: it can be a normal LTRFeaturesTargets for a baseline ranker, or a TrainExperiment if we want
        to predict the scores
        :param n:
        """
        super().__init__()
        self.ltr = ltr
        self.n = n
        if not isinstance(ltr, LTRFeaturesTargets):
            self._require_training()
        else:
            print('Received LTR Features targets. Training and scoring are not necessary')

    def _require_training(self):
        """
        Checks if the provided system has already been trained. If not, it will train it.
        :return:
        """
        if not os.path.exists(self.ltr.model_path):
            if isinstance(self.ltr, LTRBEINFTrain):
                raise ValueError("Beinf model provided hasn't been trained. Please train using R")
            else:
                # This only works for models trained with sklearn in Python
                print('Empty model. Starting train stage...')
                self.ltr.train()
        else:
            print('Model already trained')

    def config(self) -> Dict:
        pass

    def experiment_id(self) -> str:
        experiment_hash = super().experiment_id()
        return experiment_hash

    @property
    @abstractmethod
    def rank_type(self) -> str:
        pass

    @property
    def path(self) -> str:
        return '{}/{}/{}'.format(MODELS_PATH, self.rank_type, self.experiment_id())

    def load_data(self) -> Union[pd.DataFrame, Tuple[csr_matrix, pd.DataFrame]]:
        """Reads data from all matches"""
        # Baseline
        if isinstance(self.ltr, LTRFeaturesTargets):
            return self.ltr.read()
        else:
            if isinstance(self.ltr.ltr, LTRFeaturesTargetsTF):
                return self.ltr.ltr.read_features_targets()
            else:
                return self.ltr.ltr.read()

    def _write_summaries(self, df: pd.DataFrame):
        self._create_directory_if_not_exists()
        path_2_write = f'{self.path}/summaries.csv'
        print('Saving to', path_2_write)
        df.to_csv(path_2_write, index=False)

    def read_summaries(self):
        path_2_read = f'{self.path}/summaries.csv'
        print('Reading ranked summaries from', path_2_read)
        return pd.read_csv(path_2_read)

    @staticmethod
    @abstractmethod
    def rank(df: pd.DataFrame):
        pass
