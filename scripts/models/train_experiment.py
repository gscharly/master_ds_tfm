# Scripts imports
from scripts.experiments.experiment import Experiment
from scripts.conf import MODELS_PATH

# DS imports
import pandas as pd
from sklearn.pipeline import Pipeline

# Other imports
from abc import abstractmethod
from typing import Dict, List
import pickle
import os


class TrainExperiment(Experiment):
    def __init__(self, cv: int):
        super().__init__()
        self.cv = cv

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
    def train_data(self) -> pd.DataFrame:
        """Returns train data as a pandas df"""
        pass

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame):
        """Optionally preprocess data before training"""
        pass

    @property
    @abstractmethod
    def target_col(self) -> str:
        pass

    @property
    @abstractmethod
    def features_cols(self) -> List[str]:
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
        # Load and preprocess data
        df_train = self.train_data()
        df_train = self.preprocess_data(df_train)
        X_train = df_train.loc[:, self.features_cols]
        y_train = df_train[self.target_col]
        # Load model
        pipeline = self.pipeline()
        # Train model
        print('Training model...')
        pipeline.fit(X_train, y_train)
        # Persist
        model_info = self._model_info(pipeline)
        self._persist_model(pipeline, model_info)
