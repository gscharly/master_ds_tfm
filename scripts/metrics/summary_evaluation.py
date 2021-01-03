from scripts.text.basic_text_processor import BasicTextProcessor
import scripts.conf as conf
from scripts.models.rank_with_model.rank_with_model import RankModel
from scripts.models.baseline_rank.baseline_rank import BaselineRank
from scripts.metrics.sms import SMS

import os
from os import listdir
from os.path import isfile, join
import pickle
from typing import Dict, Tuple, Union, Optional
import warnings

import pandas as pd
import numpy as np
from rouge import Rouge


class SummaryEvaluation:
    AVAILABLE_METRICS = ['rouge', 'sms']
    JOIN_COLS = ['url']
    MATCH_COLS = JOIN_COLS + ['json_file', 'article', 'events']
    SUM_COLS = JOIN_COLS + ['summary_events']

    def __init__(self, metric: str, sms_dict: Optional[Dict] = None):
        assert metric in self.AVAILABLE_METRICS, 'Available metrics: {}'.format(self.AVAILABLE_METRICS)
        print('Setting target metric to', metric)
        self.metric = metric
        self.sms = SMS(**sms_dict) if sms_dict and self.metric == 'sms' else None
        self.text_proc = BasicTextProcessor()

    def _preprocess_text_smd(self, text):
        pass

    def _preprocess_text(self, text: str) -> str:
        """
        - Remove stopwords
        - Remove punctuation
        - Remove numbers
        - Lemmatize
        :param text:
        :return:
        """
        doc = self.text_proc.nlp(text)
        sentence_list = list()
        for sentence in doc.sents:
            tokens = [token.lemma_.lower() for token in sentence if self.text_proc.token_filter_stopword(token) and
                      self.text_proc.token_filter(token) and not self.text_proc.has_numbers(token)]
            sentence_list.append(' '.join(tokens))
        return '. '.join(sentence_list)

    # TODO incluir opcion para que se haga la media solo para unas determinadas noticias
    @staticmethod
    def _avg_rouge(scores_dict: Dict) -> Dict:
        avg_scores_dict = dict()
        for _, league_dict in scores_dict.items():
            for _, scores in league_dict.items():
                if not avg_scores_dict:
                    avg_scores_dict = {k: {'f': list(), 'p': list(), 'r': list()} for k in scores[0].keys()}
                for rouge_mode, metrics in scores[0].items():
                    for metric_name, metric_value in metrics.items():
                        avg_scores_dict[rouge_mode][metric_name].append(metric_value)
        scores_dict = avg_scores_dict.copy()
        for rouge_mode, metrics in avg_scores_dict.items():
            for metric_name, metrics_list in metrics.items():
                scores_dict[rouge_mode][metric_name] = sum(metrics_list) / len(metrics_list)
        return scores_dict

    @staticmethod
    def _avg_sms(scores_dict: Dict) -> float:
        values = [val for _, league_dict in scores_dict.items() for val in league_dict.values()]
        return np.mean(values)

    def _avg_scores(self, scores_dict: Dict) -> Union[Dict, float]:
        if self.metric == 'rouge':
            return self._avg_rouge(scores_dict)
        elif self.metric == 'sms':
            return self._avg_sms(scores_dict)
        else:
            raise ValueError("Metric not available")

    @staticmethod
    def rouge(candidate: str, reference: str, rouge_mode: str = None) -> Dict:
        if not rouge_mode:
            rouge = Rouge()
        else:
            rouge = Rouge(metrics=[rouge_mode])
        score = rouge.get_scores(candidate, reference)
        return score

    def score_summaries(self, candidate: str, reference: str) -> Union[Dict, float]:
        if self.metric == 'rouge':
            return self.rouge(candidate, reference)
        elif self.metric == 'sms':
            return self.sms.calculate_sms(candidate, reference)
        else:
            raise ValueError("Metric not available")

    def evaluate(self, pd_df_summaries: pd.DataFrame, path: str, preprocess_text: bool = False,
                 summary_key: str = 'summary_events') -> Tuple[Dict, Dict]:
        main_cond = os.path.exists(path + '.pickle') and os.path.exists(path + '_avg.pickle')
        cond1 = main_cond and not preprocess_text
        cond2 = main_cond and preprocess_text

        if cond1 or cond2:
            print('Metrics already exist')
            with open(path + '.pickle', 'rb') as fp:
                scores_dict = pickle.load(fp)
            with open(path + '_avg.pickle', 'rb') as fp:
                avg_scores_dict = pickle.load(fp)
            return scores_dict, avg_scores_dict
        else:
            pd_df_summaries_na = pd_df_summaries[~pd_df_summaries[summary_key].isna()]
            n_nas = len(pd_df_summaries) - len(pd_df_summaries_na)
            if n_nas:
                print("There are {} articles with an empty summary".format(n_nas))
            print("Performing evaluation for {} articles".format(len(pd_df_summaries_na)))
            scores_dict = dict()
            for _, row in pd_df_summaries_na.iterrows():
                candidate = row[summary_key]
                reference = row['article']
                if preprocess_text:
                    candidate = self._preprocess_text(candidate)
                    reference = self._preprocess_text(reference)
                if not candidate or not reference:
                    warnings.warn('Could not perform score: empty summary')
                    continue
                else:
                    scores = self.score_summaries(candidate, reference)
                    if row['json_file'] in scores_dict.keys():
                        scores_dict[row['json_file']][row['url']] = scores
                    else:
                        scores_dict[row['json_file']] = {row['url']: scores}

            print('Writing to', path + '.pickle')
            with open(path + '.pickle', 'wb') as fp:
                pickle.dump(scores_dict, fp)

            avg_scores = self._avg_scores(scores_dict)
            print('Writing avg to', path + '_avg.pickle')
            with open(path + '_avg.pickle', 'wb') as fp:
                pickle.dump(avg_scores, fp)
            return scores_dict, avg_scores

    def evaluate_all_summaries(self, preprocess_text: bool = False):
        """
        Performs evaluation for every summary file
        :param preprocess_text:
        :return:
        """
        only_files = [f for f in listdir(conf.SUMMARY_PATH) if isfile(join(conf.SUMMARY_PATH, f)) and 'map' not in f]
        print("Evaluating the following summaries:", only_files)
        pd_matches = pd.read_csv(conf.ARTICLES_PATH)
        for f in only_files:
            print('Evaluating', f)
            pd_summaries = pd.read_csv(join(conf.SUMMARY_PATH, f))
            pd_matches_sum = pd_matches[self.MATCH_COLS].merge(pd_summaries[self.SUM_COLS], on=self.JOIN_COLS,
                                                               how='inner')
            results_path = '{}/summaries/{}/{}'.format(conf.METRICS_PATH, self.metric, f.split('.')[0])
            if preprocess_text:
                results_path += '_processed'
            print(results_path)
            _ = self.evaluate(pd_matches_sum, results_path, preprocess_text=preprocess_text)

    def bound_metrics(self, preprocess_text: bool = False):
        """
        Calculates the selected metric using directly the events as a summary. An extractive summary will never be able
        to score better recall than this result.
        :param preprocess_text:
        :return:
        """
        pd_matches = pd.read_csv(conf.ARTICLES_PATH)
        suffix = '_processed' if preprocess_text else ''
        results_path = '{}/summaries/{}/upper_bound{}'.format(conf.METRICS_PATH, self.metric, suffix)
        _ = self.evaluate(pd_matches, results_path, preprocess_text=preprocess_text, summary_key='events')

    def _exp_metrics_avg(self, scores: Union[Dict, float]) -> Union[pd.DataFrame, float]:
        """
        If using sms, the avg needs no process
        :param scores:
        :return:
        """
        if self.metric == 'rouge':
            pd_dict = {
                'metric': list(),
                'metric_type': list(),
                'value': list()
            }
            for metric, metric_dict in scores.items():
                for metric_type, metric_value in metric_dict.items():
                    pd_dict['metric'].append(metric)
                    pd_dict['metric_type'].append(metric_type)
                    pd_dict['value'].append(metric_value)
            return pd.DataFrame(pd_dict)
        else:
            return scores

    @staticmethod
    def _exp_metrics_rouge(scores_dict: Dict) -> pd.DataFrame:
        pd_dict = {
            'metric': list(),
            'metric_type': list(),
            'value': list(),
            'json_file': list(),
            'url': list()
        }

        for league_file, league_dict in scores_dict.items():
            for url, match_scores_list in league_dict.items():
                match_scores_dict = match_scores_list[0]
                for metric, metric_dict in match_scores_dict.items():
                    for metric_type, metric_value in metric_dict.items():
                        pd_dict['metric'].append(metric)
                        pd_dict['metric_type'].append(metric_type)
                        pd_dict['value'].append(metric_value)
                        pd_dict['json_file'].append(league_file)
                        pd_dict['url'].append(url)
        return pd.DataFrame(pd_dict)

    @staticmethod
    def _exp_metrics_sms(scores_dict: Dict) -> pd.DataFrame:
        pd_dict = {
            'score': list(),
            'json_file': list(),
            'url': list()
        }
        for league_file, league_dict in scores_dict.items():
            for url, score in league_dict.items():
                pd_dict['score'].append(score)
                pd_dict['json_file'].append(league_file)
                pd_dict['url'].append(url)
        return pd.DataFrame(pd_dict)

    def _exp_metrics(self, scores_dict: Dict) -> pd.DataFrame:
        if self.metric == 'rouge':
            return self._exp_metrics_rouge(scores_dict)
        elif self.metric == 'sms':
            return self._exp_metrics_sms(scores_dict)
        else:
            raise ValueError('check metric')

    def experiment_metrics_pandas(self, path: str, avg: bool) -> Union[pd.DataFrame, float]:
        with open(path, 'rb') as fp:
            scores_dict = pickle.load(fp)
        if avg:
            return self._exp_metrics_avg(scores_dict)
        else:
            return self._exp_metrics(scores_dict)

    def output_avg_metrics(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the average scores for every experiment
        :return:
        """
        results_path = '{}/summaries/{}'.format(conf.METRICS_PATH, self.metric)
        only_files = [f for f in listdir(results_path) if isfile(join(results_path, f)) and 'avg' in f]
        all_scores_df = pd.DataFrame()
        all_scores = list()
        for f in only_files:
            file_score = self.experiment_metrics_pandas(join(results_path, f), avg=True)
            experiment_name = f.split('.')[0][:-4]
            if isinstance(file_score, pd.DataFrame):
                file_score['experiment'] = experiment_name
                all_scores_df = pd.concat([all_scores_df, file_score])
            else:
                all_scores.append((experiment_name, file_score))
        if self.metric == 'sms':
            all_scores_df = pd.DataFrame(all_scores, columns=['experiment', 'score'])
        return all_scores_df

    def output_avg_bound(self) -> pd.DataFrame:
        path = '{}/summaries/{}/upper_bound_avg.pickle'.format(conf.METRICS_PATH, self.metric)
        with open(path, 'rb') as fp:
            scores_dict = pickle.load(fp)
        return self._exp_metrics_avg(scores_dict)

    @staticmethod
    def _append_metric_info(path: str, rank: Union[RankModel, BaselineRank]):
        if isinstance(rank, RankModel):
            ltr_rank = rank.ltr.ltr
        else:
            ltr_rank = rank.ltr

        if ltr_rank.target_metric == 'rouge':
            rouge_type = '{}_{}'.format(ltr_rank.metric_params['rouge_mode'],
                                        ltr_rank.metric_params['rouge_metric'])
            path = f'{path}_{rouge_type}'
        else:
            path = f'{path}_{ltr_rank.target_metric}'
        return path

    def _rank_path(self, rank: Union[RankModel, BaselineRank], preprocess_text: bool):
        """
        Creates the file name for each rank experiment.
        - For a rank using a model:
            ../summaries/METRIC/RANKTYPE_MODELTYPE_TARGETMETRIC_MODELHASH_PROCESSED
        :param rank:
        :param preprocess_text:
        :return:
        """
        main_path = f'{conf.METRICS_PATH}/summaries/{self.metric}'
        if isinstance(rank, RankModel):
            path = f'{main_path}/{rank.RANK_TYPE}_{rank.ltr.MODEL_TYPE}'
            path = self._append_metric_info(path, rank)
            path = f'{path}_{rank.ltr.experiment_id()}'
            if preprocess_text:
                path = path + '_processed'
            return path
        elif isinstance(rank, BaselineRank):
            path = f'{main_path}/{rank.RANK_TYPE}'
            path = self._append_metric_info(path, rank)
            if preprocess_text:
                path = path + '_processed'
            return path
        else:
            raise ValueError('rank must be of type: RankModel, BaselineRank')

    def evaluate_rank(self, rank: Union[RankModel, BaselineRank], **evaluate_kwargs):
        rank.run()
        pd_summaries = rank.read_summaries()
        pd_matches = pd.read_csv(conf.ARTICLES_PATH)
        pd_matches_sum = pd_matches[self.MATCH_COLS].merge(pd_summaries[self.SUM_COLS], on=self.JOIN_COLS, how='inner')
        path = self._rank_path(rank, evaluate_kwargs['preprocess_text'])
        print(f'Saving to {path}')
        return self.evaluate(pd_df_summaries=pd_matches_sum, path=path, **evaluate_kwargs)
