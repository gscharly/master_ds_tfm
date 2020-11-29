from scripts.extractive_summary.ltr.ltr_targets import LTRTargets
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank
from scripts.text.article_text_processor import ArticleTextProcessor

import pandas as pd

from typing import Dict, Optional
import os
import itertools
import warnings


class LTRTargetsRanknet(LearnToRank):
    LTR_TYPE = 'targets_ranknet'

    def __init__(self, ltr_targets: LTRTargets, processor: Optional[ArticleTextProcessor] = None):
        """
        Class that builds targets for a Ranknet model. It combines ltr_targets and creates a dataset with every
        event combination for a match.
        """
        super().__init__(processor if processor else ltr_targets.processor)
        self.ltr_targets = ltr_targets

    def config(self) -> Dict:
        return self.ltr_targets.config()

    @property
    def ltr_type(self) -> str:
        return self.LTR_TYPE

    @property
    def file_path(self) -> str:
        return '{}/{}'.format(self.path, self.LTR_TYPE)

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None):
        """
        We won't be needing this
        """
        pass

    @staticmethod
    def _event_cart_product(df: pd.DataFrame):
        """
        Performs a cross join over the events of a match
        :param df:
        :return:
        """
        cols = ['event_ix', 'sentence_ix', 'score']
        x1_cols = [f'{col}_x1' for col in cols]
        x2_cols = [f'{col}_x2' for col in cols]
        x_cols = x1_cols + x2_cols
        l = list(itertools.product(df.values.tolist(), df.values.tolist()))
        result = pd.DataFrame(list(map(lambda x: sum(x, []), l)), columns=x_cols)
        return result

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame):
        df['is_dup'] = df.apply(lambda row: 1 if row['event_ix_x1'] == row['event_ix_x2'] else 0, axis=1)
        return df[df.is_dup == 0].copy()

    @staticmethod
    def _create_labels(df: pd.DataFrame):
        def _label(row):
            if row['score_x1'] < row['score_x2']:
                return 1
            elif row['score_x1'] == row['score_x2']:
                return 0
            else:
                return -1
        df['label'] = df.apply(_label, axis=1)
        return df

    def _save_csv(self, df: pd.DataFrame):
        if os.path.exists(self.file_path):
            pd_all = pd.read_csv(self.file_path)
            pd_all = pd.concat([pd_all, df])
            pd_all.to_csv(self.file_path, index=False)
        else:
            df.to_csv(self.file_path, index=False)

    @staticmethod
    def _parse_url(url: str):
        url = url.replace('http://', '')
        url = url.replace('https://', '')
        url = url.replace('www.', '')
        url = url.replace('/', '_')
        return url

    def create_targets(self):
        """
        Saves a csv comparing events for each match. The output will be:
        - event_ix_x1
        - sentence_ix_x1
        - event_ix_x2
        - label
        - url
        - json_file
        :return:
        """
        self._create_directory_if_not_exists()
        targets = self.ltr_targets.get_targets()
        urls = targets['url'].unique()
        for url in urls:
            print(url)
            parsed_url = self._parse_url(url)
            path_2_write = f'{self.file_path}_{parsed_url}.csv'
            if os.path.exists(path_2_write):
                print(f'{path_2_write} already exists')
                continue

            targets_match = targets[targets.url == url]
            if len(targets_match) < 2:
                warnings.warn(f'The following url has less than 2 events: {url}')
                continue
            json_file = targets_match['json_file'].unique()[0]
            targets_match = targets_match.drop(['url', 'json_file'], axis=1)
            crossed_events = self._event_cart_product(targets_match)
            crossed_events_no_dup = self._drop_duplicates(crossed_events)
            labels = self._create_labels(crossed_events_no_dup)
            final_match_result = labels[['event_ix_x1', 'sentence_ix_x1', 'event_ix_x2', 'label']].copy()
            final_match_result['url'] = url
            final_match_result['json_file'] = json_file
            print(f'Saving to {path_2_write}')
            final_match_result.to_csv(path_2_write, index=False)
            del final_match_result, labels, crossed_events_no_dup, targets_match

