# Scripts
from scripts.models.train_experiment import TrainExperiment
from scripts.extractive_summary.ltr.ltr_features_targets import LTRFeaturesTargets

# DS
import pandas as pd

# Other
from typing import List, Dict


class LTRBEINFTrain(TrainExperiment):
    MODEL_TYPE = 'ltr_beinf'
    TARGET_COL = 'score'
    RANDOM_SEED = 10
    N_JOBS = 5

    def __init__(self, ltr_params: Dict, **train_exp_params):
        super().__init__(**train_exp_params)
        self.ltr_params = ltr_params
        self.ltr = LTRFeaturesTargets(**ltr_params)

    def config(self) -> Dict:
        config_dict = {
            'cv': self.cv,
            'opt_metric': self.opt_metric if self.opt_metric else '',
            'features': sorted(self.features)
        }
        config_dict.update(self.model_params)
        config_dict.update(self.ltr_params)
        return config_dict

    @property
    def model_type(self) -> str:
        return self.MODEL_TYPE

    @property
    def target_col(self) -> str:
        return self.TARGET_COL

    @property
    def features_cols(self) -> List[str]:
        return self.features

    def train_data(self) -> pd.DataFrame:
        """Returns train data as a pandas df"""
        return self.ltr.read_train()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Delete players_importance and recategorize n_players
        - Delete advantage and equalize
        - Sum length and n_stop
        - Delete sim_previous and position
        :param df:
        :return:
        """
        # Number of players
        print('Categorizing n_players...')
        df['n_players_cat'] = df['n_players'].apply(lambda x: 'no_player' if x == 0 else 'one_player'
                                                    if x == 1 else 'more_than_one_player')
        # Total length
        print('Computing new length...')
        df['total_length'] = df['length'] + df['n_stop']
        drop_cols = set(df.columns).difference(set(self.features))
        print('Dropping', drop_cols)
        df_sel = df[self.features + [self.target_col]].copy()
        return df_sel

    def pipeline(self):
        """This will be done in R"""
        pass

    def model_out(self, pipeline) -> pd.DataFrame:
        """TBD"""
        pass
