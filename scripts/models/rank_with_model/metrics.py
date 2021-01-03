from scripts.models.rank_with_model.rank_with_model import RankModel

from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

import warnings
from tqdm import tqdm
import os


class RankModelMetrics:
    """
    This class calculates ranking metrics for our RankModel algorithms.
    """
    def __init__(self, rank: RankModel):
        """
        k will be the number of events selected to perform the summary
        :param rank:
        """
        self.rank = rank
        self.k = rank.n

    @property
    def metrics_path(self) -> str:
        return f'{self.rank.path}/metrics.csv'

    def _join_scores_ground_truth(self):
        """
        - Loads inferred scores from RankModel
        - Loads events and original scores from LTRTargets
        - Joins both
        :return:
        """
        scores_df = self.rank.get_scores_df()
        ground_truth_df = self.rank.ltr.ltr.targets.get_targets()
        ground_truth_df.rename({'score': 'ground_truth'}, axis=1, inplace=True)
        drop_cols = ['json_file', 'event_ix', 'sentence_ix']
        results_df = scores_df.merge(ground_truth_df, on=['url', 'event_ix'], how='inner')
        results_df.drop(drop_cols, axis=1, inplace=True)
        if len(results_df) != len(ground_truth_df):
            warnings.warn(f'Length of original events ({len(ground_truth_df)}) is different than scored'
                          f'events ({len(scores_df)})')
        return results_df

    def _calculate_metrics(self, df: pd.DataFrame) -> float:
        """
        For now, only NDCG@k metric is calculated. The functionality can be extended by returning a dict of metrics.
        df must contain score and ground_truth columns
        :param df:
        :return:
        """
        scores = np.asarray([df['score'].tolist()])
        ground_truth = np.asarray([df['ground_truth'].tolist()])
        if ground_truth.shape == (1, 1):
            warnings.warn('This url only has one event, ignoring...')
            return -1.0
        ndcg_at_k = ndcg_score(ground_truth, scores, k=self.k)
        return ndcg_at_k

    def _metrics_df(self) -> pd.DataFrame:
        """
        Calculates rank metrics for each url
        :return:
        """
        results_df = self._join_scores_ground_truth()
        urls = results_df['url'].unique()
        url_metric_list = list()
        for url in tqdm(urls):
            url_df = results_df[results_df.url == url]
            metric = self._calculate_metrics(url_df)
            # Don't count matches with only one event
            if metric == -1.0:
                continue
            url_metric_list.append((url, metric))
            del url_df
        url_metric_df = pd.DataFrame(url_metric_list, columns=['url', f'ndcg@{self.k}'])
        url_metric_df.to_csv(self.metrics_path, index=False)
        return url_metric_df

    def get_metrics(self) -> float:
        """
        Calculates the average metric (ndcg@k) for all the urls.
        :return:
        """
        if os.path.exists(self.metrics_path):
            url_metric_df = pd.read_csv(self.metrics_path)
        else:
            url_metric_df = self._metrics_df()
        metric_list = url_metric_df[f'ndcg@{self.k}'].values
        avg_metric = np.mean(metric_list)
        return avg_metric



