# Scripts
from scripts.models.train_all_experiment import TrainAllExperiment
from scripts.extractive_summary.ltr.ltr_features_targets_tf import LTRFeaturesTargetsTF
from scripts.models.dimensionality_reduction import DimensionalityReduction

# DS
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.sparse.csr import csr_matrix

# Other
from typing import Dict, Tuple, Optional
import pickle


class LTRSVMTFTrain(TrainAllExperiment):
    """
    SVM regression model to apply to TF/TFIDF generated features
    """
    MODEL_TYPE = 'ltr_svm'
    N_JOBS = 5

    def __init__(self, ltr_params: Dict, dim_reduction_params: Optional[Dict], **train_exp_params):
        super().__init__(**train_exp_params)
        self.ltr_params = ltr_params
        self.ltr = LTRFeaturesTargetsTF(**ltr_params)
        self.dim_reduction = DimensionalityReduction(**dim_reduction_params) if dim_reduction_params else None

    def config(self) -> Dict:
        config_dict = {
            'cv': self.cv,
            'opt_metric': self.opt_metric if self.opt_metric else ''
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

    def pipeline(self) -> Pipeline:
        """Define the model's pipeline"""

        if self.cv:
            print(f'Using cv with {self.cv} folds optimizing {self.opt_metric}')
            rf = SVR()
            rf = GridSearchCV(estimator=rf, param_grid=self.model_params, scoring=self.opt_metric, cv=self.cv,
                              n_jobs=self.N_JOBS)
        else:
            rf = SVR(**self.model_params)

        if self.dim_reduction:
            steps = [('dim_reduction', self.dim_reduction.dim_reduction_pipe()),
                     ('model', rf)]
        else:
            steps = [('model', rf)]

        pipe = Pipeline(steps=steps)
        return pipe

    def model_out(self, pipeline: Pipeline) -> Dict:
        if self.dim_reduction:
            dim_reduction_stage = pipeline['dim_reduction']
            return {
                'explained_variance': dim_reduction_stage.explained_variance_ratio_.sum()
            }
        else:
            return dict()

