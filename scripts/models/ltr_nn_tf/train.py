# Scripts
from scripts.models.train_nn_experiment import TrainNNExperiment
from scripts.extractive_summary.ltr.ltr_features_targets_tf import LTRFeaturesTargetsTF
from scripts.models.dimensionality_reduction import DimensionalityReduction

# DS
import numpy as np
from scipy.sparse.csr import csr_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Other
import random as python_random
from typing import Dict, Tuple
import pickle

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)


class LTRNNTFTrain(TrainNNExperiment):
    """
    Neural network to apply to TF/TFIDF generated features
    """
    MODEL_TYPE = 'ltr_nn'
    N_JOBS = 5
    # TODO: add option to include automatic dropout layers
    MANDATORY_PARAMS = ['dense_layers', 'dense_activation', 'optimizer']
    DROPOUT_NAME = 'dropout'
    METRICS = [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]

    def __init__(self, ltr_params: Dict, **train_exp_params):
        self._check_params(train_exp_params)
        # If dropout, ensure same vector length
        self._ensure_same_length_params(train_exp_params)
        # Dimensionality reduction
        dim_reduction_params = train_exp_params.get('dim_reduction_params')
        self.dim_reduction = DimensionalityReduction(**dim_reduction_params) if dim_reduction_params else None

        super().__init__(**train_exp_params)

        self.ltr_params = ltr_params
        self.ltr = LTRFeaturesTargetsTF(**ltr_params)
        print('Optimizing {} with {}'.format(self.opt_metric, self.model_params['optimizer']))
        print(f'Epochs: {self.epochs}')
        print(f'Batch size: {self.batch_size}')

    def _check_params(self, train_exp_params: Dict):
        assert all(key_param in train_exp_params['model_params'].keys() for key_param in self.MANDATORY_PARAMS), \
            f'Model parameters required are: {self.MANDATORY_PARAMS}'

    def _ensure_same_length_params(self, train_exp_params: Dict):
        len_diff = (len(train_exp_params['model_params']['dense_layers']) -
                    len(train_exp_params['model_params'][self.DROPOUT_NAME]))
        if self.DROPOUT_NAME in train_exp_params['model_params'].keys() and len_diff != 0:
            print(f'Adding {len_diff} zeros so that params have the same length')
            for i in range(len_diff):
                train_exp_params['model_params'][self.DROPOUT_NAME].append(0)

    def config(self) -> Dict:
        config_dict = {
            'cv': self.cv,
            'opt_metric': self.opt_metric if self.opt_metric else '',
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        config_dict.update(self.model_params)
        config_dict.update(self.ltr_params)
        if self.dim_reduction:
            config_dict['dim_reduction'] = self.dim_reduction.dim_reduction
            config_dict['dim_reduction_params'] = self.dim_reduction.dim_reduction_params
        return config_dict

    @property
    def model_type(self) -> str:
        mode = self.ltr_params['mode']
        return f'{self.MODEL_TYPE}_{mode}'

    def train_data(self) -> Tuple[csr_matrix, np.array]:
        """Returns x_train and y_train"""
        paths = self.ltr.datasets_path
        print(f'Loading training data from {self.ltr.path}')
        x_train = pickle.load(open(paths['x_train'], 'rb'))
        y_train = pickle.load(open(paths['y_train'], 'rb'))
        return x_train, y_train

    def validation_data(self) -> Tuple[csr_matrix, np.array]:
        """Returns x_validation and y_validation"""
        paths = self.ltr.datasets_path
        print(f'Loading validation data from {self.ltr.path}')
        x = pickle.load(open(paths['x_validation'], 'rb'))
        y = pickle.load(open(paths['y_validation'], 'rb'))
        return x, y

    def model_out(self, pipeline):
        pass

    def _add_layers(self, model, input_dim):
        i = 0
        for n, activation in zip(self.model_params['dense_layers'], self.model_params['dense_activation']):
            if i == 0:
                model.add(layers.Dense(n, input_dim=input_dim, activation=activation))
            else:
                model.add(layers.Dense(n, activation=activation))
            i += 1

    def _add_layers_with_dropout(self, model, input_dim):
        i = 0
        for n, activation, dropout in zip(self.model_params['dense_layers'], self.model_params['dense_activation'],
                                          self.model_params['dropout']):
            if i == 0:
                model.add(layers.Dense(n, input_dim=input_dim, activation=activation))
            else:
                model.add(layers.Dense(n, activation=activation))
            if dropout:
                model.add(layers.Dropout(dropout))
            i += 1

    def build_model(self, input_dim: int):
        """NN layers definition"""
        model = keras.models.Sequential()
        if self.model_params.get('dropout'):
            self._add_layers_with_dropout(model, input_dim)
        else:
            self._add_layers(model, input_dim)

        model.add(layers.Dense(1))
        # Model compilation
        model.compile(loss=self.opt_metric, optimizer=self.model_params['optimizer'], metrics=self.METRICS)
        model.summary()
        return model

    def pipeline(self):
        model = self.build_model(input_dim=self.input_dim)
        # TODO if its necessary
        # pipe = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('estimator', model)
        # ])
        return model



