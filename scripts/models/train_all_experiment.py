# Scripts imports
from scripts.experiments.experiment import Experiment
from scripts.conf import MODELS_PATH

# DS imports
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.sparse.csr import csr_matrix

# Other imports
from abc import abstractmethod
from typing import Dict, Optional, Tuple
import pickle
import os


class TrainAllExperiment(Experiment):
    """
    This abstraction is meant to be used when features are too big to use a df (for example with a
    tf with big vocabulary). This training experiment will directly load and train a preprocessed sparse
    and compressed numpy matrix.
    """
    def __init__(self, model_params: Dict, opt_metric: Optional[str] = None, cv: int = 0):
        super().__init__()
        self.model_params = model_params
        # CV settings
        self.cv = cv
        self.opt_metric = opt_metric

    def config(self) -> Dict:
        pass

    def experiment_id(self) -> str:
        experiment_hash = super().experiment_id()
        return experiment_hash

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    def path(self) -> str:
        return '{}/{}/{}'.format(MODELS_PATH, self.model_type, self.experiment_id())

    @property
    def model_path(self) -> str:
        return '{}/ckpt.pickle'.format(self.path)

    @property
    def keras_model_path(self) -> str:
        return f'{self.path}/keras_model.h5'

    def read_model(self) -> Pipeline:
        return pickle.load(open(self.model_path, 'rb'))

    @property
    def model_info_path(self) -> str:
        return '{}/model_info.pickle'.format(self.path)

    def read_model_info(self) -> Dict:
        return pickle.load(open(self.model_info_path, 'rb'))

    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Define the model's pipeline"""
        pass

    @abstractmethod
    def model_out(self, model):
        """Returns model additional outputs on training data"""
        pass

    @abstractmethod
    def train_data(self) -> Tuple[csr_matrix, np.array]:
        """Returns x_train and y_train"""
        pass

    def _persist_model(self, best_model, model_info: Dict):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print('Writing model to', self.model_path)
        pickle.dump(best_model, open(self.model_path, 'wb'))
        print('Writing model info to', self.model_info_path)
        pickle.dump(model_info, open(self.model_info_path, 'wb'))

    def _model_info(self, pipeline: Pipeline) -> Dict:
        # model = pipeline['model'] if isinstance(pipeline, Pipeline) else pipeline
        model = pipeline['model']
        return {
            'best_score': model.best_score_ if self.cv else None,
            'best_params': model.best_params_ if self.cv else None,
            'model_out': self.model_out(pipeline)
        }

    def train(self):
        """
        Method that trains a model
        :return:
        """
        self._write_config()
        X_train, y_train = self.train_data()
        # Load model
        pipeline = self.pipeline()
        # Train model
        print('Training model...')
        pipeline.fit(X_train, y_train)
        # Persist
        model_info = self._model_info(pipeline)
        self._persist_model(pipeline, model_info)
