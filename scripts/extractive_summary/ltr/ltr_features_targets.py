from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.ltr.ltr_targets import LTRTargets
from scripts.extractive_summary.ltr.ltr_features import LTRFeatures
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank

import pandas as pd
from typing import Dict, List, Optional
import os


class LTRFeaturesTargets(LearnToRank):
    LTR_TYPE = 'features_targets'

    def __init__(self, target_metric: str, key_events: List[str], lags: List[int], metric_params: Dict,
                 count_vec_kwargs: Optional[Dict], drop_teams: bool = False, lemma: bool = False):
        """

        :param key_events:
        :param lags:
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
        self.features = LTRFeatures(key_events=key_events, lags=lags, processor=self.processor,
                                    count_vec_kwargs=self.count_vec_kwargs)
        # We store these values to be able to save different experiments
        # Text processing options
        self.drop_teams = drop_teams
        self.lemma = lemma
        # Features options
        self.key_events = key_events
        self.lags = lags
        # Target options
        self.target_metric = target_metric
        self.metric_params = metric_params

    def config(self) -> Dict:
        return {
            'key_events': self.key_events,
            'lags': self.lags,
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
        return '{}/{}.csv'.format(self.path, self.LTR_TYPE)

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a pandas dataframe containing features and targets for a match.
        :param match_dict:
        :param league_season_teams:
        :return:
        """
        target_df = self.targets.run_match(match_dict, league_season_teams=league_season_teams)
        features_df = self.features.run_match(match_dict, league_season_teams=league_season_teams)
        match_df = features_df.join(target_df, how='inner')
        return match_df

    def run_target_features(self):
        """
        Computes and saves a dataset containing both features and targets for all of the articles. It first checks
        whether features and targets with the configuration are already created.
        :return:
        """
        if os.path.exists(self.file_path):
            print('{} already exists'.format(self.file_path))
            return
        targets = self.targets.get_targets()
        features = self.features.get_features()
        pd_all = features.merge(targets, on=['url', 'json_file', 'event_ix'], how='inner')
        self._write_config()
        print('Writing to', self.file_path)
        pd_all.to_csv(self.file_path, index=False)

    def get_features_targets(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            print('{} does not exists'.format(self.file_path))
            print('Executing features and targets')
            self.run_target_features()
        else:
            print('Reading features and targets from {}'.format(self.file_path))
        return self.read()
