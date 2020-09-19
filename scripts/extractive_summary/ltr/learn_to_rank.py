from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.ltr.ltr_target_metrics import TargetMetrics
from scripts.extractive_summary.ltr.ltr_features import LTRFeatures

from scripts.conf import TEAMS, CSV_DATA_PATH

import pandas as pd
from tqdm import tqdm
from functools import reduce
import os
from typing import Dict, List, Optional


class LearnToRank:
    PATH_TO_WRITE = '{}/summaries/ltr/features_targets.csv'.format(CSV_DATA_PATH)

    def __init__(self, key_events: List[str], lags: List[int], target_metric: str = 'rouge',  drop_teams: bool = False,
                 lemma: bool = False):
        """

        :param key_events:
        :param lags:
        :param target_metric: metric that will be used to build the target. One of AVAILABLE_METRICS
        :param drop_teams:
        :param lemma:
        """

        self.target_metric = target_metric
        self.processor = ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        self.metrics = TargetMetrics(metric=target_metric, processor=self.processor)
        self.features = LTRFeatures(key_events=key_events, lags=lags, processor=self.processor)

    def match_target_features(self, match_dict: Dict, metric_params: Dict, count_vec_kwargs: Optional[Dict],
                              league_season_teams: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a pandas dataframe containing features and targets for a match
        :param match_dict:
        :param metric_params:
        :param count_vec_kwargs:
        :param league_season_teams:
        :return:
        """
        # If we use want to use the same configuration when using cosine distance as score
        if not count_vec_kwargs:
            count_vec_kwargs = metric_params

        target_df = self.metrics.get_targets_pandas(match_dict, league_season_teams=league_season_teams,
                                                    **metric_params)
        features_df = self.features.get_features_pandas(match_dict, league_season_teams=league_season_teams,
                                                        **count_vec_kwargs)
        match_df = features_df.join(target_df, how='inner')
        return match_df

    def _match_exists(self, match_url: str) -> bool:
        if os.path.exists(self.PATH_TO_WRITE):
            pd_all = pd.read_csv(self.PATH_TO_WRITE)
            return match_url in pd_all['url'].unique()
        else:
            return False

    def _write_match(self, pd_match: pd.DataFrame):
        if os.path.exists(self.PATH_TO_WRITE):
            pd_all = pd.read_csv(self.PATH_TO_WRITE)
            pd_all = pd.concat([pd_all, pd_match])
            pd_all.to_csv(self.PATH_TO_WRITE, index=False)
        else:
            pd_match.to_csv(self.PATH_TO_WRITE, index=False)

    def _non_processed_dict(self, all_files: Dict) -> Dict:
        pd_all = pd.read_csv(self.PATH_TO_WRITE) if os.path.exists(self.PATH_TO_WRITE) else None
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

    def run_all_target_features(self, metric_params: Dict, count_vec_kwargs: Optional[Dict]):
        all_files = self.processor.load_json()
        all_files_proc = self._non_processed_dict(all_files)
        print('Updated all_files')
        for season_file, season_values in tqdm(all_files_proc.items()):
            print(season_file)
            self.processor.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                # if self._match_exists(match_url):
                    # print('Match already exists')
                    # continue
                print(match_url)
                match_df = self.match_target_features(match_dict, metric_params, count_vec_kwargs)
                match_df['url'] = match_url
                match_df['json_file'] = season_file
                self._write_match(match_df)
        pd_all = pd.read_csv(self.PATH_TO_WRITE)
        return pd_all

