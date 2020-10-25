from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.ltr.ltr_targets import LTRTargets
from scripts.extractive_summary.ltr.ltr_features_tf import LTRFeaturesTF
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank

from sklearn.model_selection import train_test_split

import pandas as pd
from typing import Dict, Optional
import os
import pickle


class LTRFeaturesTargetsTF(LearnToRank):
    LTR_TYPE = 'features_targets'
    RANDOM_SEED = 10

    def __init__(self, target_metric: str, metric_params: Dict, mode: str,
                 count_vec_kwargs: Optional[Dict], drop_teams: bool, lemma: bool,
                 train_perc: float = 0.7, val_perc: float = 0.2):
        """
        :param target_metric: metric that will be used to build the target. One of AVAILABLE_METRICS
        :param drop_teams:
        :param lemma:
        """
        self.processor = ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        super().__init__(processor=self.processor)
        # If we use want to use the same configuration when using cosine distance as score
        self.count_vec_kwargs = self.metric_params if not count_vec_kwargs else count_vec_kwargs
        # Metrics and features
        self.targets = LTRTargets(metric=target_metric, metric_params=metric_params, processor=self.processor)
        self.features = LTRFeaturesTF(mode=mode, count_vec_kwargs=self.count_vec_kwargs, lemma=lemma,
                                      drop_teams=drop_teams, processor=self.processor)
        # We store these values to be able to save different experiments
        # Text processing options
        self.drop_teams = drop_teams
        self.lemma = lemma
        # Target options
        self.target_metric = target_metric
        self.metric_params = metric_params
        # Train/val/test split
        self.train_perc = train_perc
        self.val_perc = val_perc

    def config(self) -> Dict:
        return {
            'target_metric': self.target_metric,
            'drop_teams': self.drop_teams,
            'lemma': self.lemma,
            'metric_params': self.metric_params,
            'count_vec_kwargs': self.count_vec_kwargs
        }

    @property
    def ltr_type(self) -> str:
        return self.LTR_TYPE

    @property
    def file_path(self) -> str:
        """NOT USED"""
        return self.path

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None) -> pd.DataFrame:
        """NOT USED"""
        pass

    def _train_val_test_split(self, x, y) -> Dict:
        train_val_perc = round(self.train_perc + self.val_perc, 1)
        test_perc = round(1 - train_val_perc, 1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, random_state=self.RANDOM_SEED)
        new_val_perc = test_perc / train_val_perc
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=new_val_perc,
                                                          random_state=self.RANDOM_SEED)
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == x.shape[0]
        return {
            'x_train': X_train,
            'y_train': y_train,
            'x_val': X_val,
            'y_val': y_val,
            'x_test': X_test,
            'y_test': y_test
        }

    @property
    def datasets_path(self) -> Dict:
        return {name: f'{self.path}/{name}.pickle' for name in ['x_train', 'y_train',
                                                                'x_val', 'y_val',
                                                                'x_test', 'y_test']}

    def _save_datasets(self, dataset_dict: Dict):
        print(f'Saving datasets in {self.path}')
        for name, dataset in dataset_dict.items():
            pickle.dump(dataset, open(f'{self.path}/{name}.pickle', 'wb'))

    def run_target_features(self):
        """
        Computes and saves a dataset containing both features and targets for all of the articles. It first checks
        whether features and targets with the configuration are already created.
        :return:
        """
        if os.path.exists(self.file_path):
            print('{} already exists'.format(self.file_path))
            return
        x = self.features.get_features()
        targets = self.targets.get_targets()
        y = targets['score'].values
        # Train/val/test split
        datasets_dict = self._train_val_test_split(x, y)
        self._write_config()
        self._save_datasets(datasets_dict)

    def get_features_targets(self) -> pd.DataFrame:
        pass
