# Scripts imports
from scripts.models.rank_experiment import RankExperiment
from scripts.text.article_text_processor import ArticleTextProcessor

# DS
import pandas as pd

# Other
from abc import ABC, abstractmethod
from typing import Dict


class RowNumberRank(RankExperiment, ABC):
    """
    This class provides methods to compute a rank using a row-number approach
    """

    def __init__(self, ltr, n: int):
        super().__init__(ltr=ltr, n=n)
        self.processor = ArticleTextProcessor()

    @abstractmethod
    def get_scores_df(self) -> pd.DataFrame:
        pass

    def config(self) -> Dict:
        train_config = self.ltr.config()
        train_config['n'] = self.n
        return train_config

    @staticmethod
    def rank(df: pd.DataFrame) -> pd.DataFrame:
        """
        row-number ranking: orders the events and assign an increasing number to them. If they are repeated, it
        also increases the ranking.
        :param df:
        :return:
        """
        print('Ranking events using row_number approach...')
        df_cols_copy = df.drop('event_ix', axis=1).copy()
        df_rank = df_cols_copy.groupby('url').rank(method='first', ascending=False)
        df_rank = df_rank.rename({'score': 'rank'}, axis=1)
        return df_rank

    def rank_events(self) -> pd.DataFrame:
        scored_df = self.get_scores_df()
        df_rank = self.rank(scored_df)
        df_cols_rank = scored_df.join(df_rank)
        # Keep N first events
        df_rank_fil = df_cols_rank[df_cols_rank['rank'] <= self.n]
        return df_rank_fil.sort_index()

    def _create_events_df(self) -> pd.DataFrame:
        all_files = self.processor.load_json()
        event_tuple_list = list()
        for saeson_file, season_values in all_files.items():
            for match_url, match_dict in season_values.items():
                for event_ix, event in enumerate(match_dict['events']):
                    event_tuple_list.append((match_url, event_ix, event))
        pd_events = pd.DataFrame(event_tuple_list, columns=['url', 'event_ix', 'event'])
        return pd_events

    @staticmethod
    def _ranked_summaries(rank_df: pd.DataFrame, events_df: pd.DataFrame):
        rank_events_df = rank_df.merge(events_df, on=['url', 'event_ix'], how='inner')
        match_rank_df = rank_events_df.groupby('url').agg({'event': list})
        match_rank_df = match_rank_df['event'].apply(lambda l: ' '.join(l))
        match_rank_df = match_rank_df.reset_index()\
                                     .rename({'event': 'summary_events'}, axis=1)

        return match_rank_df

    def run(self):
        """
        Computes summaries based on baseline ranking, saving the results in .csv
        :return:
        """
        rank_df = self.rank_events()
        events_df = self._create_events_df()
        match_rank_df = self._ranked_summaries(rank_df, events_df)
        self._write_summaries(match_rank_df)
