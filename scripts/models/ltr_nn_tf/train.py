# Scripts
from scripts.models.train_all_experiment import TrainALlExperiment
from scripts.extractive_summary.ltr.ltr_features_targets_tf import LTRFeaturesTargetsTF

# Other
from typing import Dict


class LTRNNTFTrain(TrainALlExperiment):
    """
    Neural netwrok to apply to TF/TFIDF generated features
    """
    MODEL_TYPE = 'ltr_nn'
    N_JOBS = 5

    def __init__(self, ltr_params: Dict, **train_exp_params):
        super().__init__(**train_exp_params)
        self.ltr_params = ltr_params
        self.ltr = LTRFeaturesTargetsTF(**ltr_params)