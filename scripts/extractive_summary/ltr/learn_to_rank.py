from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.ltr.ltr_target_metrics import TargetMetrics
from scripts.extractive_summary.ltr.ltr_features import LTRFeatures
from scripts.experiments.experiment import Experiment

from scripts.conf import TEAMS, CSV_DATA_PATH

import pandas as pd
from tqdm import tqdm
import os
from typing import Dict, List, Optional
import pickle


class LearnToRank(Experiment):
    LTR_PATH = '{}/summaries/ltr'.format(CSV_DATA_PATH)

    def __init__(self, target_metric: str, key_events: List[str], lags: List[int], metric_params: Dict,
                 count_vec_kwargs: Optional[Dict], drop_teams: bool = False, lemma: bool = False):
        """

        :param key_events:
        :param lags:
        :param target_metric: metric that will be used to build the target. One of AVAILABLE_METRICS
        :param drop_teams:
        :param lemma:
        """
        super().__init__()
        self.target_metric = target_metric
        self.processor = ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        self.metrics = TargetMetrics(metric=target_metric, processor=self.processor)
        self.features = LTRFeatures(key_events=key_events, lags=lags, processor=self.processor)
        # We store these values to be able to save different experiments
        self.target_metric = target_metric
        # Text processing options
        self.drop_teams = drop_teams
        self.lemma = lemma
        # Features options
        self.key_events = key_events
        self.lags = lags
        # Target options
        self.metric_params = metric_params
        # If we use want to use the same configuration when using cosine distance as score
        self.count_vec_kwargs = self.metric_params if not count_vec_kwargs else count_vec_kwargs

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

    def experiment_id(self) -> str:
        experiment_hash = super().experiment_id()
        return experiment_hash

    @property
    def path(self) -> str:
        return '{}/{}'.format(self.LTR_PATH, self.experiment_id())

    @property
    def target_features_path(self) -> str:
        return '{}/features_targets.csv'.format(self.path)

    @property
    def target_path(self) -> str:
        return '{}/targets.csv'.format(self.path)

    @property
    def config_path(self) -> str:
        return '{}/config.pickle'.format(self.path)

    def match_target_features(self, match_dict: Dict, league_season_teams: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a pandas dataframe containing features and targets for a match
        :param match_dict:
        :param league_season_teams:
        :return:
        """
        target_df = self.metrics.get_targets_pandas(match_dict, league_season_teams=league_season_teams,
                                                    **self.metric_params)
        features_df = self.features.get_features_pandas(match_dict, league_season_teams=league_season_teams,
                                                        **self.count_vec_kwargs)
        match_df = features_df.join(target_df, how='inner')
        return match_df

    def _match_exists(self, match_url: str) -> bool:
        if os.path.exists(self.target_features_path):
            pd_all = pd.read_csv(self.target_features_path)
            return match_url in pd_all['url'].unique()
        else:
            return False

    def _write_match(self, pd_match: pd.DataFrame):
        if os.path.exists(self.target_features_path):
            pd_all = pd.read_csv(self.target_features_path)
            pd_all = pd.concat([pd_all, pd_match])
            pd_all.to_csv(self.target_features_path, index=False)
        else:
            pd_match.to_csv(self.target_features_path, index=False)

    def _non_processed_dict(self, all_files: Dict) -> Dict:
        pd_all = pd.read_csv(self.target_features_path) if os.path.exists(self.target_features_path) else None
        processed_files = 0
        seasons_to_del = list()
        urls_to_del = list()
        if pd_all is not None:
            processed_url_list = pd_all['url'].unique()
            # Update deletion lists
            for season_file, season_values in all_files.items():
                if all(match_url in processed_url_list for match_url in season_values.keys()):
                    processed_files += len(season_values.keys())
                    # del all_files[season_file]
                    seasons_to_del.append(season_file)
                else:
                    for match_url, match_dict in season_values.items():
                        if match_url in processed_url_list:
                            # del all_files[season_file][match_url]
                            urls_to_del.append((season_file, match_url))
                            processed_files += 1
            # Delete here (it can't be done during loop)
            for season_file in seasons_to_del:
                del all_files[season_file]
            for season_file, match_url in urls_to_del:
                del all_files[season_file][match_url]
        print('{} matches have already been processed'.format(processed_files))
        return all_files

    def _write_config(self):
        if not os.path.exists(self.config_path):
            print('Writing config in {}'.format(self.config_path))
            with open(self.config_path, 'wb') as fp:
                pickle.dump(self.config(), fp)

    def run_all_target_features(self):
        """
        Computes and saves a dataset containing both features and targets for all of the articles.
        :return:
        """
        all_files = self.processor.load_json()
        all_files_proc = self._non_processed_dict(all_files)
        print('Updated all_files')
        print('Results path in {}'.format(self.target_features_path))
        self._write_config()
        for season_file, season_values in tqdm(all_files_proc.items()):
            print(season_file)
            self.processor.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                # if self._match_exists(match_url):
                    # print('Match already exists')
                    # continue
                print(match_url)
                match_df = self.match_target_features(match_dict)
                match_df['url'] = match_url
                match_df['json_file'] = season_file
                self._write_match(match_df)
        pd_all = pd.read_csv(self.target_features_path)
        return pd_all

    def read_features_targets(self):
        if os.path.exists(self.target_features_path):
            return pd.read_csv(self.target_features_path)
        else:
            raise ValueError("Features and targets have not been written yet")

