# Scripts
from scripts.models.train_experiment import TrainExperiment
from scripts.extractive_summary.ltr.ltr_features_targets import LTRFeaturesTargets

# DS
import pandas as pd

# Other
from typing import List, Dict, Optional


class LTRTrain(TrainExperiment):
    MODEL_TYPE = 'ltr_beinf'
    TARGET_COL = 'score'
    RANDOM_SEED = 10
    N_JOBS = 5

    def __init__(self, cat_features_dict: Dict[str, List], num_features: List[str], model_params: Dict,
                 ltr_params: Dict, opt_metric: Optional[str] = None, cv: int = 0):
        super().__init__(cv=cv)
        self.cat_features_dict = cat_features_dict
        self.cat_features = list(cat_features_dict.keys())
        self.num_features = num_features
        self.features = self.cat_features + self.num_features
        self.model_params = model_params
        self.ltr_params = ltr_params
        self.ltr = LTRFeaturesTargets(**ltr_params)
        # CV settings
        self.cv = cv
        self.opt_metric = opt_metric

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
